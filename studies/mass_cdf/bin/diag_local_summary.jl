# Merge the per-run outputs of diag_local.jl into the deliverable tables and figures.
#   julia --project=studies/mass_cdf studies/mass_cdf/bin/diag_local_summary.jl
# → results/local_diagnostics/mean_density_residual.csv (bins, with per-seed rows and seed-averaged rows),
#   mean_density_residual_fits.csv (per seed + averaged coefficients with scatter errors),
#   shear_shape_test.csv (cells) and shear_shape_test_fits.csv (per S-bin χ₃ slopes and variance fractions), figures.
let py = get(ENV, "DISCODJ_PYTHON", normpath(joinpath(@__DIR__, "..", "..", "..", "..", "disco-dj-fem", ".venv", "bin", "python")))
    get!(ENV, "JULIA_PYTHONCALL_EXE", py); ENV["JULIA_CONDAPKG_BACKEND"] = "Null"; get!(ENV, "JAX_PLATFORMS", "cpu")
end
using MassCDF, DataFrames, Statistics, Printf, CairoMakie
const M = MassCDF
dir = joinpath(M.STUDY_ROOT, "results", "local_diagnostics")

"Minimal CSV reader for the files written by MassCDF.write_csv (no quoted commas)."
function read_csv(path)
    lines = readlines(path); hdr = split(lines[1], ",")
    cols = [String[] for _ in hdr]
    for l in lines[2:end]
        for (i, v) in enumerate(split(l, ","))
            push!(cols[i], v)
        end
    end
    df = DataFrame()
    for (i, h) in enumerate(hdr)
        v = cols[i]
        parsed = tryparse.(Float64, v)
        df[!, h] = all(x -> x !== nothing, parsed) ? Float64.(parsed) : v
    end
    return df
end
function gather(prefix)
    fs = filter(f -> startswith(basename(f), prefix * "_") && endswith(f, ".csv") && !occursin("test_N32", f), readdir(dir; join = true))
    isempty(fs) && return DataFrame()
    vcat((read_csv(f) for f in fs)...; cols = :union)
end
bins = gather("mean_density_residual_bins"); fits = gather("mean_density_residual_fits")
cells = gather("shear_shape_test_cells"); sfits = gather("shear_shape_test_fits")
println("rows: bins $(nrow(bins)), fits $(nrow(fits)), cells $(nrow(cells)), shape fits $(nrow(sfits))")

keyb = ["ensemble", "R_f", "N", "method", "growth", "a", "eps", "width", "S_lo", "S_hi"]
sem(x) = length(x) > 1 ? std(x) / sqrt(length(x)) : NaN
binavg = combine(groupby(bins, keyb), nrow => :n_seeds, "n" => sum => :n_total, "S_over_sigma2" => mean => :S_over_sigma2,
                 "R_over_sigma4" => mean => :R_over_sigma4, "R_over_sigma4" => sem => :R_over_sigma4_se,
                 "R2_over_sigma4" => mean => :R2_over_sigma4, "R2_over_sigma4" => sem => :R2_over_sigma4_se,
                 "j4_over_sigma4" => (x -> mean(filter(isfinite, x))) => :j4_over_sigma4, "sigma" => mean => :sigma)
M.write_csv(joinpath(dir, "mean_density_residual.csv"), binavg)
M.write_csv(joinpath(dir, "mean_density_residual_per_seed.csv"), bins)
keyf = ["ensemble", "R_f", "N", "method", "growth", "a", "eps", "width", "target", "fit"]
fitavg = combine(groupby(fits, keyf), nrow => :n_seeds, "n" => sum => :n_total, "sigma" => mean => :sigma,
                 "c0" => mean => :c0, "c0" => sem => :c0_se, "c1" => mean => :c1, "c1" => sem => :c1_se, "c2" => mean => :c2, "c2" => sem => :c2_se,
                 "c3" => mean => :c3, "c3" => sem => :c3_se, "c0_se" => mean => :c0_se_fit, "c1_se" => mean => :c1_se_fit, "c2_se" => mean => :c2_se_fit,
                 "cov_c0c1" => mean => :cov_c0c1, "cov_c0c2" => mean => :cov_c0c2, "cov_c1c2" => mean => :cov_c1c2,
                 "resid_rms_over_sigma4" => mean => :resid_rms_over_sigma4, "mean_y_over_sigma4" => mean => :mean_y_over_sigma4)
M.write_csv(joinpath(dir, "mean_density_residual_fits.csv"), fitavg)
M.write_csv(joinpath(dir, "mean_density_residual_fits_per_seed.csv"), fits)
keyc = ["ensemble", "R_f", "N", "method", "growth", "a", "eps", "width", "S_lo", "S_hi", "chi3_lo", "chi3_hi"]
cellavg = combine(groupby(cells, keyc), nrow => :n_seeds, "n" => sum => :n_total, "chi3_mean" => mean => :chi3_mean, "S_over_sigma2" => mean => :S_over_sigma2,
                  "lnJ_mean" => mean => :lnJ_mean, "lnJ_resid_S_mean" => mean => :lnJ_resid_S_mean, "lnJ_resid_S_mean" => sem => :lnJ_resid_S_se,
                  "lnJ_resid_S_se" => (x -> sqrt(mean(x .^ 2) / length(x))) => :lnJ_resid_S_se_within)
M.write_csv(joinpath(dir, "shear_shape_test.csv"), cellavg)
keys_ = ["ensemble", "R_f", "N", "method", "growth", "a", "eps", "width", "S_lo", "S_hi"]
sfavg = combine(groupby(sfits, keys_), nrow => :n_seeds, "n" => sum => :n_total, "S_over_sigma2" => mean => :S_over_sigma2,
                "slope_chi3" => mean => :slope_chi3, "slope_chi3" => sem => :slope_chi3_se, "slope_chi3_se" => mean => :slope_chi3_se_fit,
                "slope_chi3_over_sigma4" => mean => :slope_chi3_over_sigma4, "slope_chi3_over_sigma3" => mean => :slope_chi3_over_sigma3, "slope_chi3_over_sigma3" => sem => :slope_chi3_over_sigma3_se, "var_R2" => mean => :var_R2, "curv_chi3" => mean => :curv_chi3, "curv_chi3" => sem => :curv_chi3_se,
                "var_resid_S" => mean => :var_resid_S, "var_resid_S_chi" => mean => :var_resid_S_chi,
                "frac_var_explained_by_chi3" => mean => :frac_var_explained_by_chi3, "frac_var_explained_by_chi3" => sem => :frac_var_explained_by_chi3_se, "std_lnJ" => mean => :std_lnJ)
M.write_csv(joinpath(dir, "shear_shape_test_fits.csv"), sfavg)

# ── digest ───────────────────────────────────────────────────────────────────
println("\n=== Diagnostic 2: R = −ln J − g D² S at |δ_L|/σ < w  (fit R/σ⁴ = c₀ + c₁ (S/σ²) + c₂ (S/σ²)²; seed-averaged, ± seed scatter) ===")
for r in eachrow(fitavg[(fitavg.fit .== "quad") .& (fitavg.target .∈ Ref(["R", "R2", "j4"])) .& (fitavg.width .∈ Ref([0.05, 0.2])), :])
    @printf("%-20s R_f=%2.0f %-12s a=%.1f ε=%.2f w=%.2f %-3s n=%2d  c0=%+.4f±%.4f c1=%+.4f±%.4f c2=%+.4f±%.4f  rms/σ⁴=%.3f  ⟨y⟩/σ⁴=%+.4f\n",
            r.ensemble, r.R_f, r.method, r.a, r.eps, r.width, r.target, r.n_seeds, r.c0, r.c0_se, r.c1, r.c1_se, r.c2, r.c2_se, r.resid_rms_over_sigma4, r.mean_y_over_sigma4)
end
println("\n=== Diagnostic 3: χ₃ dependence of R₂ = −ln J − δ_L − (13/42)δ_L² − gS within (δ_L, S) cells: fraction of its residual variance explained (all S bins) ===")
for r in eachrow(sfavg[(sfavg.S_lo .== 0.2) .& (sfavg.S_hi .== 2.0) .& (sfavg.width .== 0.05) .& (sfavg.eps .∈ Ref([0.2, 0.5, 1.0])), :])
    @printf("%-20s R_f=%2.0f %-12s a=%.1f ε=%.2f w=%.2f n=%2d  var_resid(S)=%.3e → with χ₃ %.3e : frac explained = %.4f ± %.4f\n",
            r.ensemble, r.R_f, r.method, r.a, r.eps, r.width, r.n_seeds, r.var_resid_S, r.var_resid_S_chi, r.frac_var_explained_by_chi3, r.frac_var_explained_by_chi3_se)
end
println("\n=== χ₃ slopes per S bin (4LPT, R_f=18, w=0.05): d⟨R₂⟩/dχ₃ in units of σ³ (third-order scaling) ===")
for r in eachrow(sfavg[(sfavg.method .== "4LPT") .& (sfavg.R_f .== 18) .& (sfavg.width .== 0.05) .& (sfavg.S_hi .<= 2.0) .& (sfavg.S_lo .>= 0.2) .& isfinite.(sfavg.S_over_sigma2) .& (sfavg.eps .∈ Ref([0.2, 0.5, 1.0])), :])
    @printf("  ε=%.2f S/σ²∈[%.1f,%.1f) n=%7d  d⟨R₂⟩/dχ₃ = %+.3e ± %.1e  → /σ³ = %+.4f ± %.4f   var(R₂)=%.2e frac explained by χ₃ = %.3f\n",
            r.eps, r.S_lo, r.S_hi, r.n_total, r.slope_chi3, r.slope_chi3_se, r.slope_chi3_over_sigma3, r.slope_chi3_over_sigma3_se, r.var_R2, r.frac_var_explained_by_chi3)
end

# ── figures ──────────────────────────────────────────────────────────────────
set_theme!(theme_latexfonts(); fontsize = 13)
fig = Figure(size = (1100, 480))
ax1 = Axis(fig[1, 1]; xlabel = L"S/\sigma^2 = T\!:\!T/\sigma^2", ylabel = L"\langle R_2 \mid S\rangle/\sigma^4,\ R_2 = -\ln J - \delta_L - \frac{13}{42}\delta_L^2 - \frac{2}{7}S", title = L"|\delta_L|/\sigma < 0.05,\ \mathrm{4LPT},\ R_f = 18")
d0 = binavg[(binavg.method .== "4LPT") .& (binavg.R_f .== 18) .& (binavg.width .== 0.05) .& (binavg.growth .== "Dn"), :]
for (i, ε) in enumerate(sort(unique(d0.eps)))
    d = d0[d0.eps .== ε, :]; sort!(d, :S_over_sigma2)
    errorbars!(ax1, d.S_over_sigma2, d.R2_over_sigma4, d.R2_over_sigma4_se; color = (:royalblue3, 0.3 + 0.7i / 7))
    scatterlines!(ax1, d.S_over_sigma2, d.R2_over_sigma4; color = (:royalblue3, 0.3 + 0.7i / 7), label = "ε = $ε")
end
dj = d0[d0.eps .== 1.0, :]; sort!(dj, :S_over_sigma2)
isempty(dj) || lines!(ax1, dj.S_over_sigma2, dj.j4_over_sigma4; color = :red, linestyle = :dash, label = L"\langle j_4 \mid S\rangle/\sigma^4\ (\varepsilon\to 0)")
axislegend(ax1; position = :lt, framevisible = false, labelsize = 10)
ax2 = Axis(fig[1, 2]; xlabel = L"\chi_3 = \mathrm{tr}\,T^3/(T\!:\!T)^{3/2}", ylabel = L"\langle R_2 \rangle - \mathrm{(fit\ in\ } S)", title = L"\mathrm{shape\ dependence},\ |\delta_L|/\sigma<0.05,\ \varepsilon = 1")
c0 = cellavg[(cellavg.method .== "4LPT") .& (cellavg.R_f .== 18) .& (cellavg.width .== 0.05) .& (cellavg.eps .== 1.0) .& (cellavg.growth .== "Dn"), :]
for (i, (lo, hi)) in enumerate(unique(zip(c0.S_lo, c0.S_hi)))
    d = c0[(c0.S_lo .== lo) .& (c0.S_hi .== hi), :]; sort!(d, :chi3_mean)
    errorbars!(ax2, d.chi3_mean, d.lnJ_resid_S_mean, d.lnJ_resid_S_se_within; color = (:seagreen, 0.3 + 0.7i / 6))
    scatterlines!(ax2, d.chi3_mean, d.lnJ_resid_S_mean; color = (:seagreen, 0.3 + 0.7i / 6), label = "S/σ² ∈ [$lo, $hi)")
end
hlines!(ax2, [0.0]; color = :black)
axislegend(ax2; position = :lt, framevisible = false, labelsize = 10)
save(joinpath(dir, "fig_mean_density_residual_and_shape.png"), fig; px_per_unit = 2); save(joinpath(dir, "fig_mean_density_residual_and_shape.pdf"), fig)
println("wrote tables and figure to $dir")
