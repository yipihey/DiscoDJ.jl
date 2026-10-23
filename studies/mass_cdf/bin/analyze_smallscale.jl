# Small smoothing scales / post-crossing: tables and figures from data/smallscale/part_*.jld2
#   julia --project=studies/mass_cdf studies/mass_cdf/bin/analyze_smallscale.jl [tag]
let py = get(ENV, "DISCODJ_PYTHON", normpath(joinpath(@__DIR__, "..", "..", "..", "..", "disco-dj-fem", ".venv", "bin", "python")))
    get!(ENV, "JULIA_PYTHONCALL_EXE", py); ENV["JULIA_CONDAPKG_BACKEND"] = "Null"; get!(ENV, "JAX_PLATFORMS", "cpu")
end
using MassCDF, DataFrames, Statistics, Printf, JLD2, CairoMakie
const M = MassCDF
tag = isempty(ARGS) ? "smallscale" : ARGS[1]
datadir = joinpath(M.STUDY_ROOT, "data", tag); resdir = joinpath(M.STUDY_ROOT, "results", tag); mkpath(resdir)
rows = DataFrame[]; cdf = NamedTuple[]; rho = nothing
for f in filter(f -> startswith(basename(f), "part_seed") && endswith(f, ".jld2") && !occursin("Jtet", f), readdir(datadir; join = true))
    d = load(f); push!(rows, d["rows"]); append!(cdf, d["cdf"]); global rho = d["meta"]["rho_star"]
end
rows = vcat(rows...; cols = :union)
# PM linear-response accuracy and rms(ψ_PM − ψ_4LPT) are parsed from the worker logs (the row Dicts with extra keys lost them in the DataFrame constructor)
pmlin = Dict{Tuple{Int,Float64},Float64}(); pmrel = Dict{Tuple{Int,Float64,Float64},Float64}()
for f in filter(f -> startswith(basename(f), "part_seed") && endswith(f, ".log"), readdir(datadir; join = true))
    sd = parse(Int, match(r"part_seed(\d+)_", basename(f)).captures[1])
    for l in eachline(f)
        m = match(r"^R_f=\s*([0-9.]+) PM linear response .*rms\(ψ_PM−ψ_1LPT\)/rms=([0-9.]+)", l)
        m !== nothing && (pmlin[(sd, parse(Float64, m.captures[1]))] = parse(Float64, m.captures[2]))
        m = match(r"^R_f=\s*([0-9.]+) ε=([0-9.]+) .*rms\(ψ_PM−ψ_4LPT\)/rms=([0-9.]+)", l)
        m !== nothing && (pmrel[(sd, parse(Float64, m.captures[1]), parse(Float64, m.captures[2]))] = parse(Float64, m.captures[3]))
    end
end
rows.pm_lin_rel_rms_vs_1LPT = [get(pmlin, (r.seed, r.R_f), NaN) for r in eachrow(rows)]
rows.rel_rms_pm_minus_4lpt = [get(pmrel, (r.seed, r.R_f, r.eps), NaN) for r in eachrow(rows)]
println("rows: $(nrow(rows)); seeds $(sort(unique(rows.seed))); R_f $(sort(unique(rows.R_f))); eps $(sort(unique(rows.eps)))")
M.write_csv(joinpath(resdir, "smallscale_realizations.csv"), rows)

sem(x) = length(x) > 1 ? std(x) / sqrt(length(x)) : NaN
ρlist = [0.1, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
cols = vcat([:sigma, :frac_Jle0, :minJ, :Y, :F_M1, :P_dpos, :F_pm, :F_mp, :meanJ, :meanJinv], [Symbol("F_M_rho", ρ) for ρ in ρlist],
            [Symbol("F2quad_rho", ρ) for ρ in ρlist], [Symbol("F2lin_rho", ρ) for ρ in ρlist], [:A1sigma, :series3])
keys_ = [:R_f, :eps, :method, :jtype, :variant]
summ = combine(groupby(rows, keys_), nrow => :n_seeds, [c => mean => Symbol(c, "_mean") for c in cols]..., [c => sem => Symbol(c, "_se") for c in cols]...)
summ.sigma2 = summ.sigma_mean .^ 2
summ.Y_over_sigma = summ.Y_mean ./ summ.sigma_mean
summ.series_Y_over_sigma = M.A1_EDS .+ M.A3_PRED .* summ.sigma2
sort!(summ, [:method, :jtype, :variant, :R_f, :eps])
M.write_csv(joinpath(resdir, "smallscale_summary.csv"), summ)

# paired differences on the same seeds: PM(tet) − 4LPT(tet), PM − theory, 4LPT − theory, at each threshold
function pick(method, jtype, variant)
    d = rows[(rows.method .== method) .& (rows.jtype .== jtype) .& (rows.variant .== variant), :]
    Dict((r.seed, r.R_f, r.eps) => r for r in eachrow(d))
end
pm = pick("PM", "tet", "signed"); l4 = pick("4LPT", "tet", "signed"); l2 = pick("2LPT", "tet", "signed"); l4v = pick("4LPT", "vertex", "signed")
pma = pick("PM", "tet", "abs")
out = Dict{Symbol,Any}[]
for (R_f, ε) in sort(unique(zip(rows.R_f, rows.eps)))
    ks = [k for k in keys(pm) if k[2] == R_f && k[3] == ε && haskey(l4, k)]
    isempty(ks) && continue
    n = length(ks)
    d = Dict{Symbol,Any}(:R_f => R_f, :eps => ε, :n_seeds => n, :sigma => mean(pm[k].sigma for k in ks),
        :frac_crossed_PM => mean(pm[k].frac_Jle0 for k in ks), :frac_crossed_4LPT => mean(l4[k].frac_Jle0 for k in ks),
        :frac_crossed_4LPT_vertex => mean(l4v[k].frac_Jle0 for k in ks), :minJ_PM => minimum(pm[k].minJ for k in ks),
        :pm_lin_rel_rms => mean(pm[k].pm_lin_rel_rms_vs_1LPT for k in ks), :rel_rms_pm_minus_4lpt => mean(pm[k].rel_rms_pm_minus_4lpt for k in ks),
        :Y_PM => mean(pm[k].Y for k in ks), :Y_4LPT => mean(l4[k].Y for k in ks), :Y_4LPT_vertex => mean(l4v[k].Y for k in ks),
        :Y_series => mean(pm[k].A1sigma + M.A3_PRED * pm[k].sigma^3 for k in ks), :Y_F2quad => mean(pm[k][Symbol("F2quad_rho1.0")] - 0.5 for k in ks),
        :dY_PM_minus_4LPT => mean(pm[k].Y - l4[k].Y for k in ks), :dY_PM_minus_4LPT_se => sem([pm[k].Y - l4[k].Y for k in ks]),
        :dY_PM_minus_series => mean(pm[k].Y - pm[k].A1sigma - M.A3_PRED * pm[k].sigma^3 for k in ks),
        :dY_PM_minus_series_se => sem([pm[k].Y - pm[k].A1sigma - M.A3_PRED * pm[k].sigma^3 for k in ks]),
        :dY_4LPT_minus_series => mean(l4[k].Y - pm[k].A1sigma - M.A3_PRED * pm[k].sigma^3 for k in ks))
    for ρ in ρlist
        c = Symbol("F_M_rho", ρ); t = Symbol("F2quad_rho", ρ)
        d[Symbol("FM_PM_rho", ρ)] = mean(pm[k][c] for k in ks); d[Symbol("FM_PMabs_rho", ρ)] = mean(pma[k][c] for k in ks)
        d[Symbol("FM_4LPT_rho", ρ)] = mean(l4[k][c] for k in ks); d[Symbol("FM_2LPT_rho", ρ)] = mean(l2[k][c] for k in ks); d[Symbol("FM_theory_rho", ρ)] = mean(pm[k][t] for k in ks)
        d[Symbol("dFM_PM_minus_4LPT_rho", ρ)] = mean(pm[k][c] - l4[k][c] for k in ks); d[Symbol("dFM_PM_minus_4LPT_se_rho", ρ)] = sem([pm[k][c] - l4[k][c] for k in ks])
        d[Symbol("dFM_PM_minus_theory_rho", ρ)] = mean(pm[k][c] - pm[k][t] for k in ks); d[Symbol("dFM_PM_minus_theory_se_rho", ρ)] = sem([pm[k][c] - pm[k][t] for k in ks])
        d[Symbol("dFM_4LPT_minus_theory_rho", ρ)] = mean(l4[k][c] - pm[k][t] for k in ks)
        d[Symbol("dFM_2LPT_minus_theory_rho", ρ)] = mean(l2[k][c] - pm[k][t] for k in ks)
    end
    push!(out, d)
end
comp = DataFrame(out); sort!(comp, [:R_f, :eps]); M.write_csv(joinpath(resdir, "smallscale_tail_comparison.csv"), comp)

# ensemble-mean full CDF curves per (R_f, eps, method)
cd = Dict{Symbol,Any}[]
for g in groupby(DataFrame([(R_f = c.R_f, eps = c.eps, method = c.method, jtype = c.jtype, variant = c.variant, seed = c.seed, F_M = c.F_M) for c in cdf]), [:R_f, :eps, :method, :jtype, :variant])
    FM = hcat((Float64.(x) for x in g.F_M)...)
    for (j, ρ) in enumerate(rho)
        push!(cd, Dict(:R_f => g.R_f[1], :eps => g.eps[1], :method => g.method[1], :jtype => g.jtype[1], :variant => g.variant[1], :n_seeds => size(FM, 2),
                       :rho_star => ρ, :F_M => mean(FM[j, :]), :F_M_se => size(FM, 2) > 1 ? std(FM[j, :]) / sqrt(size(FM, 2)) : NaN))
    end
end
M.write_csv(joinpath(resdir, "smallscale_cdf_curves.csv"), DataFrame(cd))

println("\n=== PM linear-response accuracy and crossed fractions ===")
for r in eachrow(comp)
    @printf("R_f=%4.1f ε=%.2f σ=%.3f n=%d  PM lin err %.4f  rms(ψ_PM−ψ_4LPT)/rms=%.4f  crossed: PM %.2e 4LPT(tet) %.2e 4LPT(vertex) %.2e  minJ_PM=%+.3f\n",
            r.R_f, r.eps, r.sigma, r.n_seeds, r.pm_lin_rel_rms, r.rel_rms_pm_minus_4lpt, r.frac_crossed_PM, r.frac_crossed_4LPT, r.frac_crossed_4LPT_vertex, r.minJ_PM)
end
println("\n=== mean threshold: Y = F_M(>ρ̄) − P(δ_L>0) ; series = A₁σ + A₃σ³ ; F2 = single-cube second-order ===")
for r in eachrow(comp)
    @printf("R_f=%4.1f ε=%.2f σ=%.3f  Y: PM %.5f 4LPT %.5f (vertex %.5f) series %.5f F2 %.5f | PM−4LPT %+.5f±%.5f  PM−series %+.5f±%.5f  4LPT−series %+.5f\n",
            r.R_f, r.eps, r.sigma, r.Y_PM, r.Y_4LPT, r.Y_4LPT_vertex, r.Y_series, r.Y_F2quad, r.dY_PM_minus_4LPT, r.dY_PM_minus_4LPT_se, r.dY_PM_minus_series, r.dY_PM_minus_series_se, r.dY_4LPT_minus_series)
end
println("\n=== low-density tail F_M(>ρ★) (fraction of mass above ρ★; 1 − F_M is the void mass fraction) ===")
for r in eachrow(comp), ρ in (0.25, 0.5, 0.75)
    @printf("R_f=%4.1f ε=%.2f σ=%.3f crossed=%.1e ρ★=%.2f  PM %.5f (|J| %.5f)  4LPT %.5f  2LPT %.5f  theory %.5f | PM−4LPT %+.5f±%.5f  PM−theory %+.5f±%.5f\n",
            r.R_f, r.eps, r.sigma, r.frac_crossed_PM, ρ, r[Symbol("FM_PM_rho", ρ)], r[Symbol("FM_PMabs_rho", ρ)], r[Symbol("FM_4LPT_rho", ρ)], r[Symbol("FM_2LPT_rho", ρ)], r[Symbol("FM_theory_rho", ρ)],
            r[Symbol("dFM_PM_minus_4LPT_rho", ρ)], r[Symbol("dFM_PM_minus_4LPT_se_rho", ρ)], r[Symbol("dFM_PM_minus_theory_rho", ρ)], r[Symbol("dFM_PM_minus_theory_se_rho", ρ)])
end

# figures
set_theme!(theme_latexfonts(); fontsize = 13)
fig = Figure(size = (1200, 900))
ax1 = Axis(fig[1, 1]; xlabel = L"\sigma", ylabel = L"Y/\sigma = [F_M(>\bar\rho) - P(\delta_L>0)]/\sigma", title = "mean threshold: breakdown of the σ-expansion")
ax2 = Axis(fig[1, 2]; xlabel = L"\sigma", ylabel = L"1 - F_M(>\rho_\star)\ \ (\mathrm{void\ mass\ fraction})", yscale = log10, title = L"\mathrm{low\!-\!density\ tail},\ \rho_\star/\bar\rho = 0.5, 0.25")
ax3 = Axis(fig[2, 1]; xlabel = "crossed fraction P(J ≤ 0) (PM)", ylabel = L"F_M^{\mathrm{PM}} - F_M^{\mathrm{4LPT}}", xscale = log10, title = "PM − 4LPT at the low-density thresholds")
ax4 = Axis(fig[2, 2]; xlabel = "crossed fraction P(J ≤ 0) (PM)", ylabel = L"F_M^{\mathrm{PM}} - F_M^{(2)}", xscale = log10, title = "PM − single-cube theory")
Rcols = Dict(4.0 => :darkred, 6.0 => :firebrick, 8.0 => :darkorange, 10.0 => :seagreen, 14.0 => :royalblue3)
σs = range(0, maximum(comp.sigma) * 1.05, length = 60)
lines!(ax1, σs, M.A1_EDS .+ M.A3_PRED .* σs .^ 2; color = :red, label = L"A_1 + A_3\sigma^2")
hlines!(ax1, [M.A1_EDS]; color = :black, linestyle = :dash)
for R_f in sort(unique(comp.R_f))
    d = comp[comp.R_f .== R_f, :]; sort!(d, :sigma)
    scatterlines!(ax1, d.sigma, d.Y_PM ./ d.sigma; color = Rcols[R_f], marker = :circle, label = "PM R_f=$(Int(R_f))")
    scatterlines!(ax1, d.sigma, d.Y_4LPT ./ d.sigma; color = Rcols[R_f], marker = :utriangle, linestyle = :dash)
    lines!(ax1, d.sigma, d.Y_F2quad ./ d.sigma; color = Rcols[R_f], linestyle = :dot)
    for (ρ, mk) in ((0.5, :circle), (0.25, :rect))
        scatterlines!(ax2, d.sigma, max.(1 .- d[!, Symbol("FM_PM_rho", ρ)], 1e-7); color = Rcols[R_f], marker = mk)
        lines!(ax2, d.sigma, max.(1 .- d[!, Symbol("FM_4LPT_rho", ρ)], 1e-7); color = Rcols[R_f], linestyle = :dash)
        lines!(ax2, d.sigma, max.(1 .- d[!, Symbol("FM_theory_rho", ρ)], 1e-7); color = Rcols[R_f], linestyle = :dot)
    end
    x = max.(d.frac_crossed_PM, 1e-7)
    for (ρ, mk) in ((0.25, :rect), (0.5, :circle), (0.75, :utriangle), (1.0, :diamond))
        scatterlines!(ax3, x, d[!, Symbol("dFM_PM_minus_4LPT_rho", ρ)]; color = Rcols[R_f], marker = mk, label = R_f == 6.0 ? "ρ★=$ρ" : nothing)
        scatterlines!(ax4, x, d[!, Symbol("dFM_PM_minus_theory_rho", ρ)]; color = Rcols[R_f], marker = mk)
    end
end
hlines!(ax3, [0.0]; color = :black); hlines!(ax4, [0.0]; color = :black)
axislegend(ax1; position = :lt, framevisible = false, labelsize = 10); axislegend(ax3; position = :lt, framevisible = false, labelsize = 10)
Label(fig[0, :], "circles: PM (tet J); dashed: 4LPT (tet J); dotted: single-cube second-order theory; colours: R_f = 4 … 14 h⁻¹Mpc", fontsize = 12)
save(joinpath(resdir, "fig_smallscale_tail.png"), fig; px_per_unit = 2); save(joinpath(resdir, "fig_smallscale_tail.pdf"), fig)
println("wrote $resdir")
