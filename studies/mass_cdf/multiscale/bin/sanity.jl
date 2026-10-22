# Sanity checks on seed 1 (L=1000, N=128, R_f=18): <J_cube> = 1, A_P == 27-cube average of A_C, vertex vs cube sector variances,
# and the leading coefficients of the one-scale ln J fit versus the EdS expansion  ln J = I1 - I1^2/2 + (4/7) I2 + ...
#   (from -ln J = δ + (13/42) δ² + (2/7) T:T with δ = -I1, T:T = (2/3) I1² - 2 I2)
include(joinpath(@__DIR__, "_env.jl"))
using .MS, DataFrames
resdir = joinpath(M.STUDY_ROOT, "results", "multiscale")
d = JLD2.load(joinpath(M.STUDY_ROOT, "data/multiscale/cubes_L1000_N128_R18_seed1.jld2"))
AC, AP = d["AC"], d["AP"]; n = size(AC, 2)
println("meanJ: ", d["meanJ"]); println("avg27 check (max|A_P - mean27(A_C)|): ", d["avg27_check"])
println("sector variances (seed 1): ", d["sec"])
rows = Dict{Symbol,Any}[]
for method in (4, 2), ε in (0.1, 0.2, 0.5, 1.0)
    J = d["Js"]["J_$(method)LPT_e$(ε)"]
    X = Matrix{Float64}(undef, n, 26); MS.design!(X, AC, AP, 1:n, ε, :inv)
    sel = findall(>(0), J); y = log.(Float64.(J[sel]))
    X1 = X[sel, 1:11]
    A = X1' * X1; b = X1' * y
    s = sqrt.(diag(A) ./ length(y)); β = ((A ./ (s * s')) \ (b ./ s)) ./ s
    # leading-terms-only fit (I1, I1^2, I2)
    Xl = X[sel, [2, 5, 3]]; βl = (Xl' * Xl) \ (Xl' * y)
    # ln J minus the expansion through second order
    res2 = y .- (X[sel, 2] .- 0.5 .* X[sel, 5] .+ (4 / 7) .* X[sel, 3])
    row = Dict{Symbol,Any}(:method => "$(method)LPT", :eps => ε, :rms_lnJ => sqrt(mean(abs2, y .- mean(y))), :rms_resid_fit => sqrt(mean(abs2, y .- X1 * β)),
                           :rms_resid_second_order => sqrt(mean(abs2, res2)))
    for (i, nm) in enumerate(MS.BASIS_ONE); row[Symbol("c_", replace(nm, "*" => "_", "^" => "p"))] = β[i]; end
    row[:lead_I1] = βl[1]; row[:lead_I1sq] = βl[2]; row[:lead_I2] = βl[3]
    push!(rows, row)
    @printf("%dLPT eps=%.1f: full-basis c(I1,I1^2,I2) = (%.4f, %.4f, %.4f); leading-only fit = (%.4f, %.4f, %.4f); rms resid fit %.2e, after 2nd-order expansion %.2e (rms lnJ %.3f)\n",
            method, ε, β[2], β[5], β[3], βl..., row[:rms_resid_fit], row[:rms_resid_second_order], row[:rms_lnJ])
end
M.write_csv(joinpath(resdir, "sanity_lnJ_coefficients.csv"), DataFrame(rows))
