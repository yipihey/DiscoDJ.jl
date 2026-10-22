# Diagnostic 1: full second-order single-cube CDF versus the measured F_M(>ρ̄).
#   julia --project=studies/mass_cdf studies/mass_cdf/bin/diag_single_cube.jl [ensembles...]
# F_M^(2)(σ) = ∫₀^∞ du p_{χ²₅}(u) Φ(2 g σ u / 15),  g = C/D² = (1 + D₂/D₁²)/2 (EdS: 2/7),
# which is the exact single-cube second-order law −ln J = δ_L + (13/42)δ_L² + g T:T with T:T = (2σ²/15)·u,
# u ~ χ²₅, integrated over the Gaussian δ_L at fixed T:T (the δ_L² term is dropped as in the user's definition).
# Δ_tide ≡ F_M^measured − F_M^(2) is formed per realization from the paired estimator
# F_M = 1/2 + Y_even (zero-threshold noise cancels) and from the raw +ε value.
let py = get(ENV, "DISCODJ_PYTHON", normpath(joinpath(@__DIR__, "..", "..", "..", "..", "disco-dj-fem", ".venv", "bin", "python")))
    get!(ENV, "JULIA_PYTHONCALL_EXE", py); ENV["JULIA_CONDAPKG_BACKEND"] = "Null"; get!(ENV, "JAX_PLATFORMS", "cpu")
end
using MassCDF, DataFrames, Statistics, Printf, LinearAlgebra, CairoMakie
erf(x::Float64) = ccall(:erf, Float64, (Float64,), x)      # libm (no extra dependency)
gamma(x::Float64) = ccall(:tgamma, Float64, (Float64,), x)
const M = MassCDF
datadir = joinpath(M.STUDY_ROOT, "data"); resdir = joinpath(M.STUDY_ROOT, "results", "local_diagnostics"); mkpath(resdir)
ens = isempty(ARGS) ? sort(filter(d -> !startswith(d, "FEM_") && !startswith(d, "test") && isdir(joinpath(datadir, d)) &&
                       any(f -> startswith(f, "part_") && endswith(f, ".jld2") && !occursin("_elements_", f), readdir(joinpath(datadir, d))), readdir(datadir))) : ARGS

Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
# χ²₅ density and the integral on a fine grid (u ∈ [0, 80], 40000 points; relative accuracy < 1e-9)
const UGRID = range(0.0, 80.0, length = 40001)
const PCHI5 = [u > 0 ? u^(1.5) * exp(-u / 2) / (2^2.5 * gamma(2.5)) : 0.0 for u in UGRID]
function F2(σ, g)
    f = PCHI5 .* Φ.(2g * σ / 15 .* UGRID)
    return sum((f[1:end-1] .+ f[2:end]) ./ 2 .* step(UGRID))
end
A3_single(g) = -(1 / (6 * sqrt(2π))) * (2g / 15)^3 * 315          # σ³ coefficient of F_M^(2): −1.158×10⁻³ in EdS
@printf("check: F2(0, 2/7) = %.12f ;  σ³ coefficient of F2 in EdS = %.6e ; A1 = %.6f vs 2g/3/√(2π) = %.6f\n",
        F2(0.0, 2 / 7), A3_single(2 / 7), M.A1_EDS, (2 / 3) * (2 / 7) / sqrt(2π))

# exact-growth g at the epochs used
dj = M.make_dj(; N = 32, L = 1000.0, R_f = 18.0)
gexact = Dict(a => (1 + M.growth(dj, a).D2 / M.growth(dj, a).D1^2) / 2 for a in (0.5, 1.0))
println("g (exact ΛCDM) = ", gexact, "   EdS 2/7 = ", 2 / 7)

out = Dict{Symbol,Any}[]
for e in ens
    rows = M.load_ensemble(joinpath(datadir, e)).rows
    for fam in groupby(rows, M.FAMILY)
        g = fam.growth[1] == "Dn" ? 2 / 7 : gexact[fam.a[1]]
        for ε in sort(unique(abs.(fam.eps)))
            p = fam[fam.eps .== ε, :]; m = fam[fam.eps .== -ε, :]
            seeds = intersect(p.seed, m.seed); isempty(seeds) && continue
            ip = Dict(r.seed => r for r in eachrow(p)); im = Dict(r.seed => r for r in eachrow(m))
            Fpair = [0.5 + (ip[s].Y + im[s].Y) / 2 for s in seeds]; Fraw = [ip[s].F_M1 for s in seeds]
            σs = [ip[s].sigma for s in seeds]
            F2s = [F2(σ, g) for σ in σs]
            Δ = Fpair .- F2s; Δraw = Fraw .- F2s
            σm = mean(σs); n = length(seeds)
            push!(out, Dict(:ensemble => e, :ic_type => fam.ic_type[1], :method => fam.method[1], :growth => fam.growth[1], :a => fam.a[1],
                :R_f => rows.R_f[1], :L => rows.L[1], :N => rows.N[1], :eps => ε, :n_seeds => n, :sigma => σm, :sigma2 => σm^2, :g => g,
                :F_M_measured_pair => mean(Fpair), :F_M_measured_pair_se => std(Fpair) / sqrt(n),
                :F_M_measured_raw => mean(Fraw), :F_M_measured_raw_se => std(Fraw) / sqrt(n),
                :F_M2 => mean(F2s), :F_M2_taylor => 0.5 + (2g / 3) / sqrt(2π) * σm + A3_single(g) * σm^3,
                :Delta_tide => mean(Δ), :Delta_tide_se => std(Δ) / sqrt(n), :Delta_tide_raw => mean(Δraw), :Delta_tide_raw_se => std(Δraw) / sqrt(n),
                :Delta_over_sigma3 => mean(Δ) / σm^3, :Delta_over_sigma3_se => std(Δ) / sqrt(n) / σm^3,
                :A3_single_cube => A3_single(g)))
        end
    end
end
df = DataFrame(out); sort!(df, [:ensemble, :ic_type, :method, :growth, :a, :eps])
M.write_csv(joinpath(resdir, "single_cube_cdf.csv"), df)

# intercept fits Δ_tide/σ³ = a₀ + a₁σ² per family (weighted by realization scatter)
fits = Dict{Symbol,Any}[]
for fam in groupby(df, [:ensemble, :ic_type, :method, :growth, :a])
    nrow(fam) >= 3 || continue
    f = M.polyfit_cov(fam.sigma2, fam.Delta_over_sigma3, [0, 1]; w = 1 ./ fam.Delta_over_sigma3_se .^ 2)
    push!(fits, Dict(:ensemble => fam.ensemble[1], :ic_type => fam.ic_type[1], :method => fam.method[1], :growth => fam.growth[1], :a => fam.a[1],
                     :R_f => fam.R_f[1], :N => fam.N[1], :L => fam.L[1], :n_seeds => fam.n_seeds[1], :g => fam.g[1],
                     :A3_tide_intercept => f.coef[1], :A3_tide_intercept_se => sqrt(f.cov[1, 1]), :slope_sigma2 => f.coef[2], :slope_se => sqrt(f.cov[2, 2]),
                     :chi2 => sum((f.resid .^ 2) ./ fam.Delta_over_sigma3_se .^ 2), :dof => nrow(fam) - 2,
                     :A3_implied => f.coef[1] + fam.A3_single_cube[1], :A3_tide_expected_if_A3_is_1p6e3 => 1.6e-3 - fam.A3_single_cube[1]))
end
fdf = DataFrame(fits); M.write_csv(joinpath(resdir, "single_cube_cdf_fits.csv"), fdf)
println("\n=== Δ_tide/σ³ = a₀ + a₁σ²: intercept a₀ ≈ A₃,tide = A₃ − A₃,single-cube (expected ≈ 2.8×10⁻³ if A₃ = 1.6×10⁻³) ===")
for r in eachrow(fdf[(r -> r.method in ("2LPT", "3LPT", "4LPT", "5LPT", "6LPT", "3LPTx")).(eachrow(fdf)), :])
    @printf("%-22s %-8s %-5s a=%.1f g=%.5f n=%3d  A3_tide = %+.5f ± %.5f  (→ A3 = %+.5f)  slope=%+.3f±%.3f χ²/dof=%.1f/%d\n", r.ensemble, r.ic_type, r.method, r.a, r.g, r.n_seeds,
            r.A3_tide_intercept, r.A3_tide_intercept_se, r.A3_implied, r.slope_sigma2, r.slope_se, r.chi2, r.dof)
end

# figure
set_theme!(theme_latexfonts(); fontsize = 13)
fig = Figure(size = (900, 560))
ax = Axis(fig[1, 1]; xlabel = L"\sigma^2", ylabel = L"\Delta_{\mathrm{tide}}/\sigma^3 = [F_M^{\mathrm{meas}} - F_M^{(2)}]/\sigma^3", title = "single-cube second-order CDF residual")
cols = Dict("gaussian" => :royalblue3, "fixed" => :darkorange2)
main = "L1000_N128_R18" in ens ? "L1000_N128_R18" : ens[1]
for (meth, mk) in (("4LPT", :circle), ("3LPTx", :utriangle)), ic in ("gaussian", "fixed")
    d = df[(df.ensemble .== main) .& (df.method .== meth) .& (df.ic_type .== ic) .& (df.a .== 1.0), :]
    isempty(d) && continue
    errorbars!(ax, d.sigma2, d.Delta_over_sigma3, d.Delta_over_sigma3_se; color = cols[ic])
    scatterlines!(ax, d.sigma2, d.Delta_over_sigma3; color = cols[ic], marker = mk, label = "$meth $ic")
end
hlines!(ax, [1.6e-3 - A3_single(2 / 7)]; color = :red, linestyle = :dash, label = L"A_3 - A_{3,\mathrm{single}} = 2.76\times10^{-3}")
hlines!(ax, [0.0]; color = :black)
axislegend(ax; position = :lt, framevisible = false)
save(joinpath(resdir, "fig_single_cube_cdf_residual.png"), fig; px_per_unit = 2); save(joinpath(resdir, "fig_single_cube_cdf_residual.pdf"), fig)
println("wrote $resdir")
