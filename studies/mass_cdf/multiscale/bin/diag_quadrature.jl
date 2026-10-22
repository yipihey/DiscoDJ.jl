# Diagnostic: is the parent's gain a deterministic finite-h (quadrature) correction?
# At linear order the Kuhn-cube volume uses vertex finite differences, J_lin(k) = -δ_k Σ_a k̂_a² sinc(k_a h/2) Π_{b≠a} cos(k_b h/2),
# whereas tr A_C(k) = -δ_k Σ_a k̂_a² Π_b sinc(k_b h/2)  (cube average) and tr A_P with sinc(3 k h/2).
# Gaussian box sums give the linear-level residual variance of J_lin regressed on tr A_C alone and on (tr A_C, tr A_P).
include(joinpath(@__DIR__, "_env.jl"))
using .MS, DataFrames
resdir = joinpath(M.STUDY_ROOT, "results", "multiscale")
function linear_level(R_f, N, L; kind = :lcdm)
    dj0 = M.make_dj(; N, L, R_f); kt, Pt = M.pk_table(dj0)
    h = L / N; kf = 2π / L
    kx = kf .* [i <= N ÷ 2 ? i : i - N for i in 0:N-1]; kz = kf .* collect(0:N÷2)
    G = zeros(3, 3)       # <c_X c_Y> for X,Y ∈ (J, C, P)
    for iz in eachindex(kz), iy in 1:N, ix in 1:N
        (ix - 1 == N ÷ 2 || iy - 1 == N ÷ 2 || iz - 1 == N ÷ 2) && continue
        k = (kx[ix], kx[iy], kz[iz]); k2 = sum(abs2, k); k2 == 0 && continue
        w = iz == 1 ? 1.0 : 2.0
        P = M.linterp(kt, Pt, sqrt(k2))
        s1 = [MS.sincx(k[a] * h / 2) for a in 1:3]; s3 = [MS.sincx(3 * k[a] * h / 2) for a in 1:3]; co = [cos(k[a] * h / 2) for a in 1:3]
        cC = sum(k[a]^2 / k2 * prod(s1) for a in 1:3)
        cP = sum(k[a]^2 / k2 * prod(s3) for a in 1:3)
        cJ = sum(k[a]^2 / k2 * s1[a] * prod(co[b] for b in 1:3 if b != a) for a in 1:3)
        c = (cJ, cC, cP)
        for a in 1:3, b in 1:3; G[a, b] += w * P * c[a] * c[b]; end
    end
    G ./= L^3
    VJ = G[1, 1]
    VC = VJ - G[1, 2]^2 / G[2, 2]
    Gcp = G[2:3, 2:3]; g = G[1, 2:3]
    VCP = VJ - dot(g, Gcp \ g)
    return (VJ = VJ, VC = VC, VCP = VCP, R_gain_lin = 1 - VCP / VC, unexpl_one = VC / VJ, unexpl_two = VCP / VJ, slope_C = G[1, 2] / G[2, 2],
            coef = Gcp \ g)
end
rows = Dict{Symbol,Any}[]
for R in (14.0, 18.0, 22.0), (N, L) in ((128, 1000.0),)
    r = linear_level(R, N, L)
    @printf("R_f=%g N=%d: linear-level  var(J_lin)=%.3e  resid(C)=%.3e (%.2e of var)  resid(C,P)=%.3e (%.2e)  R_gain_lin=%.3f  coefs on (trA_C,trA_P)=(%.3f,%.3f)\n",
            R, N, r.VJ, r.VC, r.unexpl_one, r.VCP, r.unexpl_two, r.R_gain_lin, r.coef...)
    push!(rows, Dict{Symbol,Any}(:R_f => R, :N => N, :L => L, :var_Jlin => r.VJ, :resid_one => r.VC, :resid_two => r.VCP, :unexpl_one => r.unexpl_one,
                                 :unexpl_two => r.unexpl_two, :R_gain_linear_level => r.R_gain_lin, :coef_C => r.coef[1], :coef_P => r.coef[2]))
end
# measured analogue on seed 1 at small amplitude (ln J on [1, I1C] versus [1, I1C, I1P]), all cubes
d = JLD2.load(joinpath(M.STUDY_ROOT, "data/multiscale/cubes_L1000_N128_R18_seed1.jld2")); AC, AP = d["AC"], d["AP"]; n = size(AC, 2)
for ε in (0.1, 0.2, 0.5)
    J = d["Js"]["J_4LPT_e$ε"]
    I1C = ε .* (Float64.(AC[1, :]) .+ Float64.(AC[2, :]) .+ Float64.(AC[3, :])); I1P = ε .* (Float64.(AP[1, :]) .+ Float64.(AP[2, :]) .+ Float64.(AP[3, :]))
    y = log.(Float64.(J))
    X1 = [ones(n) I1C]; X2 = [ones(n) I1C I1P]
    r1 = y .- X1 * (X1 \ y); r2 = y .- X2 * (X2 \ y)
    c2 = X2 \ y
    @printf("seed 1, eps=%.1f, R_f=18: linear-only fit lnJ ~ 1 + I1C: resid var %.3e ; + I1P: %.3e ; R_gain %.3f ; coefs (I1C, I1P) = (%.4f, %.4f)\n", ε, var(r1), var(r2), 1 - var(r2) / var(r1), c2[2], c2[3])
    push!(rows, Dict{Symbol,Any}(:R_f => 18.0, :N => 128, :L => 1000.0, :eps => ε, :measured_linear_only_resid_one => var(r1), :measured_linear_only_resid_two => var(r2),
                                 :measured_R_gain_linear_only => 1 - var(r2) / var(r1), :coef_C => c2[2], :coef_P => c2[3]))
end
df = vcat([DataFrame(r) for r in rows]...; cols = :union)
for c in names(df); df[!, c] = coalesce.(df[!, c], NaN); end
M.write_csv(joinpath(resdir, "diag_quadrature.csv"), df)
