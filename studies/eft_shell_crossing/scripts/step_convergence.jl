# Time-step convergence of the production PM (BullFrog, uniform in D) at fixed realization:
#
#   julia --project=. scripts/step_convergence.jl conv_pm256m2bf steps_pm256m2bf_n8 steps_pm256m2bf_n16 ...
#
# The first config is the reference (200 steps).  For every variant and common epoch D, with the IR-member linear
# field x = D ψ₁∥ as regressor (same realization ⇒ the short×short noise cancels in every difference):
#   δC₂     = C₂(variant) − C₂(ref)                     k ≤ 0.08, one-term fit of R = −C₂k²
#   δC₂,δC₄ of the two-term fit (k ≤ 0.18)
#   departure = |C₂(ref) − C₂(7LPT)|   (conv_lpt256, the quantity the study measures; 0 before crossing)
#   δψ      = rms over stored modes (k ≤ 0.25) of ψ_UV(variant) − ψ_UV(ref), relative to rms ψ_UV(ref)
#   δP/P    = max over k-bins ≤ 0.25 of the UV density power ratio − 1
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, Printf, Statistics, JSON3
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
key(r, m, w) = only(k for k in keys(r) if k[1] == m && k[2] == w && k[3] == 1 && k[5] == 1)
pk = planck_pk()
cref, rref = load_config(joinpath(scratch(), ARGS[1]))
clpt, rlpt = load_config(joinpath(scratch(), get(ENV, "LPT_CONFIG", "conv_lpt256")))
L = cref["L"]; kf = 2π / L; edges = collect(0.5:1:(cref["k_split"] / kf)) .* kf
nvec = rref[key(rref, "pm", :IR)]["nvec"]; kvec = kf .* nvec; km = vec(sqrt.(sum(abs2, kvec; dims=2)))
sel = km .< cref["k_split"]
ψ1 = linear_psi1_modes(cref, 1, 1, nvec, pk)[sel, :]
pedges = collect(0.5:1:(0.25 / kf)) .* kf

function fits(dψ, D)
    x = D .* longitudinal(ψ1, kvec[sel, :]); y = longitudinal(dψ[sel, :], kvec[sel, :])
    A = accum!(BinAcc(length(edges) - 1), y, x, km[sel], kbin_index(km[sel], edges)); r = response(A)
    σ = sqrt.(max.(A.yy .- r.R .* A.yx, 0) ./ A.n ./ (2 .* A.xx))
    return fit_counterterm(r.k2, r.k4, r.R, σ; kmax=0.08), fit_counterterm(r.k2, r.k4, r.R, σ; kmax=0.18, order=2)
end
pmΔ(r, e) = ComplexF64.(r[key(r, "pm", :UV)]["psi"][e, :, :] .- r[key(r, "pm", :IR)]["psi"][e, :, :])
lptΔ(e, D) = sum(D^m .* ComplexF64.(rlpt[key(rlpt, "lpt", :UV)]["psi"][m, :, :] .- rlpt[key(rlpt, "lpt", :IR)]["psi"][m, :, :]) for m in 1:7)
function pmax_dev(δa, δb)
    bi = kbin_index(km, pedges); worst = 0.0
    for b in 1:length(pedges) - 1
        m = bi .== b; any(m) || continue
        worst = max(worst, abs(sum(abs2, δa[m]) / sum(abs2, δb[m]) - 1))
    end
    worst
end
eref = Float64.(cref["epochs"]); elpt = Float64.(clpt["epochs"])
out = Dict{String,Any}("reference" => ARGS[1], "ref_steps" => length(rref[key(rref, "pm", :UV)]["grid"]) - 1,
                       "ref_seconds" => sum(r["seconds"] for r in values(rref)))
@printf "reference %s: %d steps, %.0f s (UV+IR)\n" ARGS[1] out["ref_steps"] out["ref_seconds"]
for c in ARGS[2:end]
    cv, rv = load_config(joinpath(scratch(), c))
    ns = length(rv[key(rv, "pm", :UV)]["grid"]) - 1; secs = sum(r["seconds"] for r in values(rv))
    @printf "\n%s: %d steps, %.0f s (UV+IR)\n" c ns secs
    println("    D   departure   δC2(k≤.08)  δC2/dep   | δC2(2-term)  δC4(2-term) [C4 ref]  | rms δψ/ψ   max|δP/P|")
    tab = Dict{String,Any}("steps" => ns, "seconds" => secs, "rows" => Any[])
    for (e, D) in enumerate(Float64.(cv["epochs"]))
        D <= cv["pm"]["D_ini"] && continue
        er = findfirst(≈(D), eref); el = findfirst(≈(D), elpt); er === nothing && continue
        f1v, f2v = fits(pmΔ(rv, e), D); f1r, f2r = fits(pmΔ(rref, er), D); f1l, _ = fits(lptΔ(el, D), D)
        dep = abs(f1r.C2 - f1l.C2)
        ψv = ComplexF64.(rv[key(rv, "pm", :UV)]["psi"][e, :, :]); ψr = ComplexF64.(rref[key(rref, "pm", :UV)]["psi"][er, :, :])
        dpsi = sqrt(sum(abs2, ψv .- ψr) / sum(abs2, ψr))
        dP = pmax_dev(ComplexF64.(rv[key(rv, "pm", :UV)]["dens"][e, :]), ComplexF64.(rref[key(rref, "pm", :UV)]["dens"][er, :]))
        dc2 = f1v.C2 - f1r.C2
        @printf "%6.2f  %9.4f   %+10.5f  %8.4f   | %+10.5f  %+10.4f [%+8.3f] | %8.2e  %8.2e\n" D dep dc2 abs(dc2) / max(dep, 1e-12) (f2v.C2 - f2r.C2) (f2v.C4 - f2r.C4) f2r.C4 dpsi dP
        push!(tab["rows"], Dict("D" => D, "departure" => dep, "dC2" => dc2, "dC2_2" => f2v.C2 - f2r.C2,
                                "dC4_2" => f2v.C4 - f2r.C4, "C2_ref" => f1r.C2, "C4_ref" => f2r.C4, "dpsi_rel" => dpsi, "dP_max" => dP))
    end
    out[c] = tab
end
open(io -> JSON3.write(io, out), joinpath(@__DIR__, "..", "results", "step_convergence.json"), "w")
