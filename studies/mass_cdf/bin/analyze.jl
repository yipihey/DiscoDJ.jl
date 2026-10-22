# Aggregate worker outputs of one LPT ensemble into the machine-readable result tables.
#   julia --project=studies/mass_cdf studies/mass_cdf/bin/analyze.jl <ensemble_name> [datadir] [resultsdir]
# Produces results/<ensemble>/: realization_summary.csv, ensemble_summary.csv, cdf_curves.csv,
# shear_bins.csv, amplitude_fit.csv (+ amplitude_fit_cov.txt, amplitude_fit_per_seed.csv),
# odd_even.csv, yodd_line_fit.csv, m4_conditioning.csv, meta.toml
let py = get(ENV, "DISCODJ_PYTHON", normpath(joinpath(@__DIR__, "..", "..", "..", "..", "disco-dj-fem", ".venv", "bin", "python")))
    get!(ENV, "JULIA_PYTHONCALL_EXE", py); ENV["JULIA_CONDAPKG_BACKEND"] = "Null"; get!(ENV, "JAX_PLATFORMS", "cpu")
end
using MassCDF, DataFrames, Statistics, Printf, TOML, JLD2, LinearAlgebra
const M = MassCDF
name = ARGS[1]
datadir = length(ARGS) >= 2 ? ARGS[2] : joinpath(M.STUDY_ROOT, "data", name)
resdir = length(ARGS) >= 3 ? ARGS[3] : joinpath(M.STUDY_ROOT, "results", name)
mkpath(resdir)
E = M.load_ensemble(datadir)
rows = E.rows
println("loaded $(nrow(rows)) rows, $(length(unique(rows.seed))) seeds, ic types $(unique(rows.ic_type)), methods $(unique(rows.method))")
rho_star = E.meta[1]["rho_star"]

# 1. realization-level table (one row per realization × method × epoch × amplitude)
sort!(rows, [:ic_type, :method, :growth, :a, :seed, :eps])
M.write_csv(joinpath(resdir, "realization_summary.csv"), rows)

# 2. ensemble summary per family × ε
statcols = [:sigma, :P_dpos, :F_M1, :Y, :meanJ, :meanJinv, :mean_lnJ, :minJ, :frac_Jle0, :min_sv, :F_pp, :F_pm, :F_mp, :F_mm,
            Symbol.(filter(n -> startswith(n, "J_pct_") || startswith(n, "F_M_rho") || startswith(n, "F_V_rho"), names(rows)))...]
summ = M.group_mean_se(rows, [M.FAMILY..., :eps], statcols)
extra = combine(groupby(rows, [M.FAMILY..., :eps]), :minJ => minimum => :minJ_min, :frac_Jle0 => maximum => :frac_Jle0_max,
                :sigma_box => first => :sigma_box, :sigma_cont => first => :sigma_cont, :Y => std => :Y_std_realization)
summ = innerjoin(summ, extra; on = [M.FAMILY..., :eps])
summ.sum_transitions = summ.F_pp_mean .+ summ.F_pm_mean .+ summ.F_mp_mean .+ summ.F_mm_mean
summ.A1sigma_EdS = M.A1_EDS .* summ.sigma_mean
sort!(summ, [:ic_type, :method, :growth, :a, :eps])
M.write_csv(joinpath(resdir, "ensemble_summary.csv"), summ)

# 3. CDF curves
M.write_csv(joinpath(resdir, "cdf_curves.csv"), M.cdf_curves_table(E.cdf, rho_star))

# 4. shear bins
sh = E.shear
shs = combine(groupby(sh, [:ic_type, :method, :growth, :a, :eps, :width]), nrow => :n_seeds, :n => sum => :n_total, :n_excl => sum => :n_excl_total,
              :slope => mean => :slope_mean, :slope => (x -> std(x) / sqrt(length(x))) => :slope_se,
              :intercept => mean => :intercept_mean, :intercept => (x -> std(x) / sqrt(length(x))) => :intercept_se,
              :rms => mean => :rms_mean, :sigma => mean => :sigma_mean)
shs.slope_pred_EdS .= 2 / 7
sort!(shs, [:ic_type, :method, :growth, :a, :eps, :width])
M.write_csv(joinpath(resdir, "shear_bins.csv"), shs)
M.write_csv(joinpath(resdir, "shear_bins_per_seed.csv"), sh)

# 5. amplitude fits, odd/even, line fit
af = M.amplitude_fits(rows)
M.write_csv(joinpath(resdir, "amplitude_fit.csv"), af.summary)
M.write_csv(joinpath(resdir, "amplitude_fit_per_seed.csv"), af.per_seed)
M.write_csv(joinpath(resdir, "odd_even.csv"), af.oddeven)
lf = M.yodd_line_fit(af.oddeven)
M.write_csv(joinpath(resdir, "yodd_line_fit.csv"), lf)
open(joinpath(resdir, "amplitude_fit_cov.txt"), "w") do io
    for (k, v) in sort(collect(af.cov); by = first)
        println(io, "# $k   powers = $(v["powers"])   sigma0 = $(v["sigma0"])")
        println(io, "## covariance of the ensemble-mean coefficients (realization scatter / n)")
        for r in eachrow(v["cov_of_mean"]); println(io, join((@sprintf("%.6e", x) for x in r), " ")); end
        if v["gls_cov"] !== nothing
            println(io, "## GLS covariance (fit to the ensemble-mean curve with empirical ε–ε covariance)")
            for r in eachrow(v["gls_cov"]); println(io, join((@sprintf("%.6e", x) for x in r), " ")); end
        end
        println(io)
    end
end
jldsave(joinpath(resdir, "amplitude_fit_cov.jld2"); cov = af.cov)

# 6. m4 conditioning
if nrow(E.mcond) > 0
    M.write_csv(joinpath(resdir, "m4_conditioning.csv"), M.m4_extrapolation(E.mcond))
    M.write_csv(joinpath(resdir, "m4_conditioning_per_seed.csv"), E.mcond)
end

# 7. meta
meta = Dict("ensemble" => name, "n_parts" => length(E.meta), "sigma2_box" => E.meta[1]["sigma2_box"], "sigma2_cont" => E.meta[1]["sigma2_cont"],
            "k0_cutoff" => E.meta[1]["k0_cutoff"], "A1_EdS" => M.A1_EDS, "A1_exact_a1" => E.meta[1]["A1_exact_a1"],
            "config" => E.meta[1]["config"], "commits" => Dict(k => E.meta[1][k] for k in ("commit_DiscoDJ.jl", "commit_disco-dj-fem")),
            "seeds_per_ic_type" => Dict(string(k) => length(unique(rows.seed[rows.ic_type .== k])) for k in unique(rows.ic_type)),
            "total_volume_per_ic_type_Gpc3" => Dict(string(k) => length(unique(rows.seed[rows.ic_type .== k])) * (rows.L[1] / 1000)^3 for k in unique(rows.ic_type)))
tomlsafe(x) = x === nothing ? "none" : x isa Dict ? Dict(string(k) => tomlsafe(v) for (k, v) in x) : x isa NamedTuple ? Dict(string(k) => v for (k, v) in pairs(x)) : x
open(joinpath(resdir, "meta.toml"), "w") do io; TOML.print(io, tomlsafe(meta)); end

# console digest
println("\n=== $name : A1, A3 from per-seed odd fits (c_k/σ0^k; σ0 = ensemble-mean σ at ε=1) ===")
for r in eachrow(af.summary[(af.summary.fit .== "O") .& (af.summary.growth .== "Dn") .& (af.summary.a .== 1.0), :])
    @printf("%-9s %-5s power %d  n=%3d  A=%+.6f ± %.6f (jk %.6f, bs %.6f)   GLS A=%+.6f ± %.6f  χ²/dof=%.1f/%d\n", r.ic_type, r.method, r.power, r.n_seeds,
            r.A_mean, r.A_se, r.c_jackknife / r.sigma0^r.power, r.c_bootstrap / r.sigma0^r.power, r.A_gls, r.A_gls_se, r.chi2_gls, r.dof_gls)
end
println("predictions: A1_EdS = $(M.A1_EDS)   A1_exact(a=1) = $(E.meta[1]["A1_exact_a1"])   A3_pred ≈ $(M.A3_PRED)")
println("\n=== even coefficients (unconstrained paired fit), 4LPT, Dn ===")
for r in eachrow(af.summary[(af.summary.fit .== "U") .& (af.summary.method .== "4LPT") .& (af.summary.growth .== "Dn"), :])
    @printf("%-9s power %d  c=%+.3e ± %.3e  (%.1fσ)   A=%+.5f ± %.5f\n", r.ic_type, r.power, r.c_mean, r.c_se, r.c_mean / r.c_se, r.A_mean, r.A_se)
end
println("\n=== paired Y_even = [F_M(+ε)+F_M(−ε)]/2 − 1/2 (signal) and Y_odd (noise, expect 0), 4LPT Dn ===")
for r in eachrow(af.oddeven[(af.oddeven.method .== "4LPT") .& (af.oddeven.growth .== "Dn"), :])
    @printf("%-9s ε=%.2f σ=%.4f  Y_even=%+.6f ± %.6f   Y_odd=%+.2e ± %.2e (%.1fσ)   Y_even/σ=%.5f ± %.5f  [A1+A3σ²=%.5f]\n", r.ic_type, r.eps, r.sigma, r.Y_even, r.Y_even_se,
            r.Y_odd, r.Y_odd_se, r.Y_odd / r.Y_odd_se, r.Y_pair_over_sigma, r.Y_pair_over_sigma_se, M.A1_EDS + M.A3_PRED * r.sigma2)
end
println("\n=== Y_even/σ = A1 + A3 σ² line fits ===")
for r in (nrow(lf) > 0 ? eachrow(lf[(lf.method .∈ Ref(["2LPT", "3LPT", "4LPT", "3LPTx"])), :]) : [])
    @printf("%-9s %-5s a=%.1f %-4s %s = %+.6f ± %.6f  χ²/dof=%.1f/%d\n", r.ic_type, r.method, r.a, r.fit, r.coef, r.value, r.se, r.chi2, r.dof)
end
if nrow(E.mcond) > 0
    println("\n=== conditional moments ⟨j_k | δ_L=0⟩/σ^k (w→0 extrapolation; pred m2 = 4/21 = $(4/21), m4 ≈ $(M.M4_PRED)) ===")
    mx = M.m4_extrapolation(E.mcond)
    for g in groupby(mx, [:ic_type, :variant, :k])
        r = g[1, :]
        @printf("%-9s %-4s k=%d  extrap(w²→0) = %+.6f ± %.6f   quad-intercept = %+.6f ± %.6f   [w=0.05: %+.6f ± %.6f, w=0.2: %+.6f ± %.6f]\n", r.ic_type, r.variant, r.k,
                r.extrap_w2, r.extrap_w2_se, r.quad_intercept_mean, r.quad_intercept_se,
                g.mean[findfirst(==(0.05), g.width)], g.se[findfirst(==(0.05), g.width)], g.mean[findfirst(==(0.2), g.width)], g.se[findfirst(==(0.2), g.width)])
    end
end
# compress the large per-realization tables (git-friendly; `gunzip` to read)
for f in ("realization_summary.csv", "shear_bins_per_seed.csv", "m4_conditioning_per_seed.csv", "amplitude_fit_per_seed.csv")
    p = joinpath(resdir, f)
    isfile(p) && filesize(p) > 5_000_000 && run(`gzip -f $p`)
end
println("\nwrote tables to $resdir")
