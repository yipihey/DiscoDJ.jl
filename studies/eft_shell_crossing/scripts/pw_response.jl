# Plane-wave (sparse long-mode) response:  julia --project=. scripts/pw_response.jl <config> [...]
#
# Long field = s·ε Σ_j cos(k_j·x) (config [planewaves]), short band as in the study.  For every plane wave j:
#   Δ_odd(k_j) = [ψ_UV(+ε) − ψ_UV(−ε)]/2 − x(k_j),     x = D ψ₁,PW   (the IR member of a plane wave is exact
#                                                                  Zel'dovich: no self-coupling, no crossing)
#   R_j = Re(Δ_odd∥ x∥*) / |x∥|²,   C₂ = −R/k² (+ C₄ k² …)
# If the IR member was run, its own odd response (PM vs exact ZA) is reported as a check.
# Noise floor: the same odd combination at stored modes that carry no plane wave, within ±1.5 k_f of |k_j|
# (leakage of the plane waves through short×short pairs + PM chaos), expressed as an equivalent σ_R.
include(joinpath(@__DIR__, "..", "src", "EFTShell.jl"))
using .EFTShell, Printf, Statistics, JSON3
scratch() = get(ENV, "EFTSC_SCRATCH", "/zpool/nvme/data/tabel_scratch/eft_shell_crossing")
function getrun(r, w, sg; model="pm")
    ks = [k for k in keys(r) if k[1] == model && k[2] == w && k[5] == sg]
    return isempty(ks) ? nothing : r[only(ks)]
end

out = Dict{String,Any}()
for name in ARGS
    cfg, runs = load_config(joinpath(scratch(), name)); pk = planck_pk()
    L = cfg["L"]; kf = 2π / L; amp = cfg["planewaves"]["amp"]
    up = getrun(runs, :UV, 1); um = getrun(runs, :UV, -1); ir = getrun(runs, :IR, 1)
    nvec = up["nvec"]; kvec = kf .* nvec; km = vec(sqrt.(sum(abs2, kvec; dims=2)))
    ψ1 = linear_psi1_modes(cfg, 1, 1, nvec, pk)
    pws = [Tuple(Int.(n)) for n in cfg["planewaves"]["n"]]
    idx = Int[]; for n in pws
        i = findfirst(m -> Tuple(nvec[m, :]) == n || Tuple(nvec[m, :]) == .-n, axes(nvec, 1))
        i === nothing && error("plane wave $n not among the stored modes (k_store)"); push!(idx, i)
    end
    off = [m for m in axes(nvec, 1) if !(m in idx) && km[m] > 0 && any(abs(km[m] - km[i]) < 1.5kf for i in idx)]
    steps = length(up["grid"]) - 1; secs = sum(r["seconds"] for r in values(runs))
    @printf "\n%s: ε = %.3g, %d plane waves, %d steps, %.0f s total\n" name amp length(pws) steps secs
    println("    D  |  −R_j/k_j² for k_j = ", join([@sprintf("%.4f", km[i]) for i in idx], "  "))
    tab = Dict{String,Any}("amp" => amp, "k" => km[idx], "n" => pws, "steps" => steps, "seconds" => secs, "rows" => Any[])
    for (e, D) in enumerate(Float64.(cfg["epochs"]))
        D <= cfg["pm"]["D_ini"] && continue
        x = D .* longitudinal(ψ1, kvec)
        Δ = longitudinal((ComplexF64.(up["psi"][e, :, :]) .- ComplexF64.(um["psi"][e, :, :])) ./ 2, kvec) .- x
        R = [real(Δ[i] * conj(x[i])) / abs2(x[i]) for i in idx]
        σR = sqrt(mean(abs2, Δ[off]) / 2) / mean(abs.(x[idx]))       # per-mode noise in R units (typical |x_j|)
        Rir = ir === nothing ? fill(NaN, length(idx)) :
              [real((longitudinal(ComplexF64.(ir["psi"][e, :, :]), kvec)[i] - x[i]) * conj(x[i])) / abs2(x[i]) for i in idx]
        @printf "%5.2f | %s | σ_R(noise) %.1e  → σ(C₂) at k_1: %.4f | IR-member R (PM vs ZA): %s\n" D join([@sprintf("%+8.4f", -R[j] / km[idx[j]]^2) for j in eachindex(idx)], " ") σR σR / km[idx[1]]^2 join([@sprintf("%.1e", r) for r in Rir], " ")
        # same initial conditions evolved with nLPT (if run): PM − LPT_n per wave, realization-matched
        lu = getrun(runs, :UV, 1; model="lpt"); lm = getrun(runs, :UV, -1; model="lpt"); li = getrun(runs, :IR, 1; model="lpt")
        Rl = Dict{Int,Vector{Float64}}()
        if lu !== nothing
            for n in (1, 3, 5, 7)
                dl(r) = sum(D^m .* ComplexF64.(r["psi"][m, :, :]) for m in 1:n)
                Δl = longitudinal((dl(lu) .- dl(lm)) ./ 2, kvec) .- x
                Rl[n] = [real(Δl[i] * conj(x[i])) / abs2(x[i]) for i in idx]
            end
            @printf "      | LPT1..7 −R/k² at k_1: %s | PM − 7LPT: %s" join([@sprintf("%+8.4f", -Rl[n][1] / km[idx[1]]^2) for n in (1, 3, 5, 7)], " ") join([@sprintf("%+8.4f", -(R[j] - Rl[7][j]) / km[idx[j]]^2) for j in eachindex(idx)], " ")
            li === nothing || @printf " | IR 7LPT vs ZA: %.1e" real((longitudinal(sum(D^m .* ComplexF64.(li["psi"][m, :, :]) for m in 1:7), kvec)[idx[1]] - x[idx[1]]) * conj(x[idx[1]])) / abs2(x[idx[1]])
            println()
        end
        push!(tab["rows"], Dict("D" => D, "R" => R, "R_lpt" => Dict(string(n) => v for (n, v) in Rl), "sigma_R" => σR, "R_ir" => [isfinite(r) ? r : nothing for r in Rir]))
    end
    out[name] = tab
end
open(io -> JSON3.write(io, out), joinpath(@__DIR__, "..", "results", "pw_response_" * join(ARGS, "+") * ".json"), "w")
