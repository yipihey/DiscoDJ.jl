# Realization-level PM − LPT residual response for single-realization convergence configs:
#   ΔR(k) = Σ Re[(Δψ_PM − Δψ_LPTn)_∥ x*] / Σ|x|²  →  C₂(PM) − C₂(LPTn) with the common short×short
#   noise cancelled.   julia --project=. scripts/residual_response.jl <lpt_config> <pm_config> [...]
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, Printf, Statistics, JSON3
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
cl, rl = load_config(joinpath(scratch(), ARGS[1])); pk = planck_pk()
L = cl["L"]; kf = 2π / L; edges = collect(0.5:1:(cl["k_split"] / kf)) .* kf
key(r, m, w) = only(k for k in keys(r) if k[1] == m && k[2] == w && k[3] == 1 && k[5] == 1)
nvec = rl[key(rl, "lpt", :IR)]["nvec"]; kvec = kf .* nvec; km = vec(sqrt.(sum(abs2, kvec; dims=2)))
sel = km .< cl["k_split"]; kvec = kvec[sel, :]; km = km[sel]
ψ1 = linear_psi1_modes(cl, 1, 1, nvec, pk)[sel, :]
epochs = Float64.(cl["epochs"])
function fit(y, x)   # per-bin σ_R² = ⟨|y − R x|²⟩ / (2 Σ|x|²)  (complex Gaussian residual)
    A = accum!(y, x, km, edges); r = response(A)
    σ = sqrt.(max.(A.yy .- r.R .* A.yx, 0) ./ A.n ./ (2 .* A.xx))
    return fit_counterterm(r.k2, r.k4, r.R, σ; kmax=0.08), A
end
for pmc in ARGS[2:end]
    cp_, rp = load_config(joinpath(scratch(), pmc)); @assert rp[key(rp, "pm", :IR)]["nvec"] == nvec
    println("\n$pmc  (C₂ residual PM − LPTn, Mpc²/h², kmax = 0.08; mode-counting σ of the residual)")
    println("   D     n=1      n=3      n=5      n=7     σ(n=7)  coherence(PM−LPT7 vs x)")
    tab = Dict{String,Any}("D" => Float64[], "inv_pm" => Float64[], "pm_config" => cp_["pm"], "seconds" =>
                           sum(r["seconds"] for r in values(rp)), "lpt_config" => ARGS[1])
    for n in (1, 3, 5, 7); tab["resid_$n"] = Float64[]; tab["sigma_$n"] = Float64[]; end
    for (e, D) in enumerate(epochs)
        D <= cp_["pm"]["D_ini"] && continue
        push!(tab["D"], D); push!(tab["inv_pm"], rp[key(rp, "pm", :UV)]["inv"][e])
        dpm = ComplexF64.(rp[key(rp, "pm", :UV)]["psi"][e, :, :] .- rp[key(rp, "pm", :IR)]["psi"][e, :, :])[sel, :]
        x = D .* longitudinal(ψ1, kvec)
        @printf "%.3f " D
        for n in (1, 3, 5, 7)
            dl = sum(D^m .* ComplexF64.(rl[key(rl, "lpt", :UV)]["psi"][m, :, :] .- rl[key(rl, "lpt", :IR)]["psi"][m, :, :]) for m in 1:n)[sel, :]
            y = longitudinal(dpm .- dl, kvec)
            f, A = fit(y, x)
            @printf " %8.4f" f.C2
            push!(tab["resid_$n"], f.C2); push!(tab["sigma_$n"], f.σC2)
            n == 7 && @printf "  %7.4f  %6.3f" f.σC2 sum(A.yx) / sqrt(sum(A.yy) * sum(A.xx))
        end
        println()
    end
    out = joinpath(@__DIR__, "..", "results", "residuals_$(pmc).json"); mkpath(dirname(out))
    open(io -> JSON3.write(io, tab), out, "w")
end
