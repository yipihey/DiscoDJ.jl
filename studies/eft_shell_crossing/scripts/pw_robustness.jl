# Robustness of the plane-wave counterterm fits (from results/pw_<cfg>.h5; no raw data needed)
#
#   julia --project=. scripts/pw_robustness.jl prod_ks12 prod_ks15 ...   → results/pw_robustness.json
#
# For PM and for the departure PM − 7LPT at D = 0.5, 0.7, 1:
#   (a) fit range: two-term fit R = −C₂k² + C₄k⁴ on the waves with k ≤ f·k_max(waves), f = 1, 0.8, 0.65
#   (b) model:     two-term vs three-term (+ C₆k⁶) on all waves
#   (c) χ²/dof of the two-term fit with per-wave errors from the realization scatter
#   (d) nuisance constant: R = R₀ − C₂k² + C₄k⁴ (a k-independent R₀ is forbidden by translation invariance, so a
#       non-zero R₀ is a numerical error on the long wave; fitting it out is an alternative estimator)
# Errors: delete-one jackknife over realizations.
using HDF5, Printf, Statistics, JSON3

function fitw(k, R, order)
    A = hcat([(-1)^p .* k .^ (2p) for p in 1:order]...)
    return A \ R
end
fitr0(k, R) = hcat(ones(length(k)), -k .^ 2, k .^ 4) \ R
function jk(f, R4)      # R4: (wave, seed); f(Rmean) → vector
    n = size(R4, 2); full = f(vec(mean(R4; dims=2)))
    parts = [f(vec(mean(R4[:, setdiff(1:n, i)]; dims=2))) for i in 1:n]; m = sum(parts) ./ n
    return full, sqrt.((n - 1) / n .* sum(p -> (p .- m) .^ 2, parts))
end

out = Dict{String,Any}()
println("cfg       D   series    C₂ (f=1)          C₄ (f=1)        C₄ (f=0.8)      C₄ (f=0.65)     C₂,C₄ 3-term              χ²/dof   | with nuisance R₀")
for c in ARGS
    h = h5open(joinpath(@__DIR__, "..", "results", "pw_$c.h5")); R = read(h["R"]); k = read(h["k"]); E = read(h["epochs"]); close(h)
    o = sortperm(k); k = k[o]; R = R[:, :, o, :]
    res = Dict{String,Any}()
    for D in (0.5, 0.7, 1.0), (lab, sel) in (("PM", r -> r[1, :, :, :]), ("PM-7LPT", r -> r[1, :, :, :] .- r[8, :, :, :]))
        e = findfirst(≈(D), E); X = sel(R)[e, :, :]
        row = Dict{String,Any}()
        print(rpad(c, 9), @sprintf("%.1f  %-8s", D, lab))
        for (i, f) in enumerate((1.0, 0.8, 0.65))
            m = k .<= f * maximum(k) .+ 1e-12
            v, s = jk(r -> fitw(k[m], r, 2), X[m, :])
            row["f$f"] = Dict("C2" => v[1], "C4" => v[2], "sC2" => s[1], "sC4" => s[2], "nwaves" => count(m))
            i == 1 ? @printf("  %+8.4f ± %.4f  %+7.2f ± %5.2f", v[1], s[1], v[2], s[2]) : @printf("  %+7.2f ± %5.2f", v[2], s[2])
        end
        v3, s3 = jk(r -> fitw(k, r, 3), X)
        row["three_term"] = Dict("C2" => v3[1], "C4" => v3[2], "C6" => v3[3], "sC2" => s3[1], "sC4" => s3[2], "sC6" => s3[3])
        v0, s0 = jk(r -> fitr0(k, r), X)
        row["r0_fit"] = Dict("R0" => v0[1], "C2" => v0[2], "C4" => v0[3], "sR0" => s0[1], "sC2" => s0[2], "sC4" => s0[3])
        Rm = vec(mean(X; dims=2)); σ = vec(std(X; dims=2)) ./ sqrt(size(X, 2))
        p = fitw(k, Rm, 2); χ2 = sum(abs2, (Rm .- hcat(-k .^ 2, k .^ 4) * p) ./ σ) / (length(k) - 2)
        row["chi2dof"] = χ2
        @printf("  %+8.4f/%+7.2f ± %.4f/%5.2f  %6.2f   | R₀-fit: C₂ %+8.4f ± %.4f  C₄ %+7.2f ± %5.2f  R₀ %+.1e\n", v3[1], v3[2], s3[1], s3[2], χ2, v0[2], s0[2], v0[3], s0[3], v0[1])
        res["$(lab)_D$D"] = row
    end
    out[c] = res
end
open(io -> JSON3.write(io, out), joinpath(@__DIR__, "..", "results", "pw_robustness.json"), "w")
