# Figures for the mass-CDF study.
#   julia --project=studies/mass_cdf studies/mass_cdf/bin/make_figures.jl <main_ensemble> [convergence_ensembles...]
# Reads results/<ensemble>/*.csv-equivalent tables from the JLD2/DataFrames produced by analyze.jl
# (re-derived here from the raw worker outputs for the element-level panels) and writes results/figures/*.png|pdf.
let py = get(ENV, "DISCODJ_PYTHON", normpath(joinpath(@__DIR__, "..", "..", "..", "..", "disco-dj-fem", ".venv", "bin", "python")))
    get!(ENV, "JULIA_PYTHONCALL_EXE", py); ENV["JULIA_CONDAPKG_BACKEND"] = "Null"; get!(ENV, "JAX_PLATFORMS", "cpu")
end
using MassCDF, DataFrames, Statistics, Printf, JLD2, CairoMakie, LinearAlgebra
const M = MassCDF
main = ARGS[1]
conv = length(ARGS) >= 2 ? ARGS[2:end] : String[]
figdir = joinpath(M.STUDY_ROOT, "results", "figures"); mkpath(figdir)
E = M.load_ensemble(joinpath(M.STUDY_ROOT, "data", main))
rows = E.rows
af = M.amplitude_fits(rows)
oe = af.oddeven
rho_star = E.meta[1]["rho_star"]
A1 = M.A1_EDS; A3p = M.A3_PRED
colors = Dict("gaussian" => :royalblue3, "fixed" => :darkorange2)
set_theme!(theme_latexfonts(); fontsize = 14)
save_fig(name, fig) = (save(joinpath(figdir, name * ".png"), fig; px_per_unit = 2); save(joinpath(figdir, name * ".pdf"), fig); println("  wrote $name"))

# ── Fig 1: Y_odd/σ versus σ² — straight line with intercept A1 and slope A3 ─────────────
fig = Figure(size = (900, 620))
ax = Axis(fig[1, 1]; xlabel = L"\sigma^2", ylabel = L"Y_{\mathrm{even}}/\sigma = [F_M(+\varepsilon)+F_M(-\varepsilon)]/(2\sigma) - 1/(2\sigma)", title = "$main: 4LPT, D^n growth (a = 1)")
for ic in sort(unique(oe.ic_type))
    d = oe[(oe.ic_type .== ic) .& (oe.method .== "4LPT") .& (oe.growth .== "Dn"), :]
    sort!(d, :sigma2)
    errorbars!(ax, d.sigma2, d.Y_pair_over_sigma, d.Y_pair_over_sigma_se; color = colors[ic], whiskerwidth = 8)
    scatter!(ax, d.sigma2, d.Y_pair_over_sigma; color = colors[ic], markersize = 11, label = "$ic ICs ($(d.n_seeds[1]) seeds)")
end
xs = range(0, maximum(oe.sigma2) * 1.05, length = 50)
lines!(ax, xs, fill(A1, length(xs)); color = :black, linestyle = :dash, label = L"A_1 = 4/(21\sqrt{2\pi})")
lines!(ax, xs, A1 .+ A3p .* xs; color = :red, label = L"A_1 + 1.6\times10^{-3}\,\sigma^2\ \mathrm{(prediction)}")
lf = M.yodd_line_fit(oe)
for ic in (nrow(lf) > 0 ? sort(unique(oe.ic_type)) : String[])
    d = lf[(lf.ic_type .== ic) .& (lf.method .== "4LPT") .& (lf.growth .== "Dn") .& (lf.fit .== "line"), :]
    nrow(d) == 2 || continue
    a1 = d.value[d.coef .== "A1"][1]; a3 = d.value[d.coef .== "A3"][1]; a3e = d.se[d.coef .== "A3"][1]
    lines!(ax, xs, a1 .+ a3 .* xs; color = colors[ic], linestyle = :dot, label = @sprintf("fit %s: A_3 = %.2e ± %.2e", ic, a3, a3e))
end
axislegend(ax; position = :lt, framevisible = false, labelsize = 11)
save_fig("fig1_Yodd_over_sigma_vs_sigma2", fig)

# ── Fig 2: odd and even paired-amplitude components versus ε ─────────────────────────────
fig = Figure(size = (1000, 460))
ax1 = Axis(fig[1, 1]; xlabel = L"\varepsilon", ylabel = L"Y_{\mathrm{even}}(\varepsilon) - A_1\sigma(\varepsilon)", title = "paired (even) part minus leading order")
ax2 = Axis(fig[1, 2]; xlabel = L"\varepsilon", ylabel = L"Y_{\mathrm{odd}}(\varepsilon)", title = "odd part (zero-threshold noise, predicted 0)")
for ic in sort(unique(oe.ic_type))
    d = oe[(oe.ic_type .== ic) .& (oe.method .== "4LPT") .& (oe.growth .== "Dn"), :]; sort!(d, :eps)
    errorbars!(ax1, d.eps, d.Y_even .- A1 .* d.sigma, d.Y_even_se; color = colors[ic], whiskerwidth = 8)
    scatterlines!(ax1, d.eps, d.Y_even .- A1 .* d.sigma; color = colors[ic], label = ic)
    errorbars!(ax2, d.eps, d.Y_odd, d.Y_odd_se; color = colors[ic], whiskerwidth = 8)
    scatterlines!(ax2, d.eps, d.Y_odd; color = colors[ic], label = ic)
end
d = oe[(oe.ic_type .== first(sort(unique(oe.ic_type)))) .& (oe.method .== "4LPT") .& (oe.growth .== "Dn"), :]; sort!(d, :eps)
lines!(ax1, d.eps, A3p .* d.sigma .^ 3; color = :red, label = L"A_3\sigma^3,\ A_3 = 1.6\times10^{-3}")
hlines!(ax1, [0.0]; color = :black, linestyle = :dash); hlines!(ax2, [0.0]; color = :black, linestyle = :dash)
axislegend(ax1; position = :lt, framevisible = false, labelsize = 11); axislegend(ax2; position = :lt, framevisible = false, labelsize = 11)
save_fig("fig2_odd_even_vs_eps", fig)

# ── Fig 3: transition fractions F_{+-} and F_{-+} versus σ ───────────────────────────────
summ = M.group_mean_se(rows[(rows.growth .== "Dn") .& (rows.eps .> 0), :], [:ic_type, :method, :eps], [:sigma, :F_pm, :F_mp, :F_pp, :F_mm, :Y])
fig = Figure(size = (1000, 460))
ax1 = Axis(fig[1, 1]; xlabel = L"\sigma", ylabel = "fraction", title = L"F_{+-} = P(\delta_L<0,\,J<1)\quad F_{-+} = P(\delta_L>0,\,J>1)")
ax2 = Axis(fig[1, 2]; xlabel = L"\sigma", ylabel = L"F_{-+}/F_{+-}", title = "ratio (shear focusing dominance)")
for (mi, meth) in enumerate(["2LPT", "4LPT"])
    for ic in sort(unique(summ.ic_type))
        d = summ[(summ.ic_type .== ic) .& (summ.method .== meth), :]; sort!(d, :sigma_mean)
        ls = meth == "4LPT" ? :solid : :dash
        scatterlines!(ax1, d.sigma_mean, d.F_pm_mean; color = colors[ic], linestyle = ls, marker = :circle, label = "F₊₋ $meth $ic")
        scatterlines!(ax1, d.sigma_mean, d.F_mp_mean; color = colors[ic], linestyle = ls, marker = :utriangle, label = "F₋₊ $meth $ic")
        scatterlines!(ax2, d.sigma_mean, d.F_mp_mean ./ d.F_pm_mean; color = colors[ic], linestyle = ls, label = "$meth $ic")
    end
end
σs = range(0, maximum(summ.sigma_mean) * 1.05, length = 50)
lines!(ax1, σs, A1 .* σs; color = :black, linestyle = :dot, label = L"A_1\sigma")
axislegend(ax1; position = :lt, framevisible = false, labelsize = 10); axislegend(ax2; position = :lt, framevisible = false, labelsize = 10)
save_fig("fig3_transition_fractions", fig)

# ── Fig 4: shear focusing near δ_L = 0 : −ln J versus T:T ────────────────────────────────
efiles = filter(f -> occursin("_elements_seed1.jld2", f) && startswith(basename(f), "part_gaussian"), readdir(joinpath(M.STUDY_ROOT, "data", main); join = true))
fig = Figure(size = (1100, 480))
ax1 = Axis(fig[1, 1]; xlabel = L"T\!:\!T", ylabel = L"-\ln J", title = L"|\delta_L|/\sigma < 0.05,\ \varepsilon = 1\ \mathrm{(seed\ 1)}")
ax2 = Axis(fig[1, 2]; xlabel = L"\varepsilon", ylabel = "regression slope", title = L"-\ln J\ \mathrm{vs}\ T\!:\!T\ \mathrm{slope}\ (\to 2/7)")
if !isempty(efiles)
    el = load(efiles[1])
    δ = el["delta_L"]; σ0 = el["sigma0"]; TT = el["TT"]
    m = abs.(δ) .< 0.05f0 * σ0
    for (key, col, lab) in (("J_1LPT_eps1.0", :gray60, "1LPT"), ("J_2LPT_eps1.0", :seagreen, "2LPT"), ("J_4LPT_eps1.0", :royalblue3, "4LPT"))
        haskey(el, key) || continue
        y = -log.(Float64.(el[key][m])); x = Float64.(TT[m])
        idx = 1:max(1, length(x) ÷ 4000):length(x)
        scatter!(ax1, x[idx], y[idx]; color = (col, 0.35), markersize = 4, label = lab)
    end
    xs = range(0, maximum(TT[m]), length = 20)
    lines!(ax1, xs, (2 / 7) .* xs; color = :red, label = L"\frac{2}{7}\,T\!:\!T")
    haskey(el, "j2") && lines!(ax1, xs, (2 / 7) .* xs .+ (13 / 42) * (0.05 * σ0)^2 / 3; color = :red, linestyle = :dot)
    axislegend(ax1; position = :lt, framevisible = false, labelsize = 11)
end
sh = E.shear
shs = combine(groupby(sh[(sh.ic_type .== "gaussian") .& (sh.growth .== "Dn") .& (sh.eps .> 0) .& (sh.method .∈ Ref(["1LPT", "2LPT", "4LPT"])), :], [:method, :eps, :width]),
              :slope => mean => :slope, :slope => (x -> std(x) / sqrt(length(x))) => :se)
for (meth, col) in (("1LPT", :gray50), ("2LPT", :seagreen), ("4LPT", :royalblue3)), (w, mk) in ((0.02, :circle), (0.2, :utriangle))
    d = shs[(shs.method .== meth) .& (shs.width .== w), :]; sort!(d, :eps)
    isempty(d) && continue
    errorbars!(ax2, d.eps, d.slope, d.se; color = col)
    scatterlines!(ax2, d.eps, d.slope; color = col, marker = mk, label = "$meth, |δ|/σ<$w")
end
hlines!(ax2, [2 / 7]; color = :red, linestyle = :dash, label = "2/7")
j2 = sh[(sh.method .== "j2exact") .& (sh.ic_type .== "gaussian"), :]
if nrow(j2) > 0
    d = combine(groupby(j2, :width), :slope => mean => :slope); sort!(d, :width)
    hlines!(ax2, [d.slope[1]]; color = :black, linestyle = :dot, label = @sprintf("exact j₂ regression (w=%.2f): %.4f", d.width[1], d.slope[1]))
end
ylims!(ax2, 0.27, 0.52)
axislegend(ax2; position = :lt, framevisible = false, labelsize = 10)
save_fig("fig4_shear_focusing", fig)

# ── Fig 5: full F_M(>ρ★) curves: 2LPT, 4LPT (main ensemble) and FEM (L=500 runs) ────────
cdf = M.cdf_curves_table(E.cdf, rho_star)
fig = Figure(size = (1100, 760))
ax1 = Axis(fig[1, 1]; xscale = log10, xlabel = L"\rho_\star/\bar\rho", ylabel = L"F_M(>\rho_\star)", title = "mass CDF, ε = 1 (Gaussian ICs)")
ax2 = Axis(fig[2, 1]; xscale = log10, xlabel = L"\rho_\star/\bar\rho", ylabel = L"\Delta F_M\ \mathrm{relative\ to\ 4LPT}", title = "differences")
for (meth, col, ls) in (("1LPT", :gray50, :dot), ("2LPT", :seagreen, :dash), ("3LPT", :purple, :dashdot), ("4LPT", :royalblue3, :solid))
    d = cdf[(cdf.ic_type .== "gaussian") .& (cdf.method .== meth) .& (cdf.eps .== 1.0) .& (cdf.growth .== "Dn"), :]; sort!(d, :rho_star)
    isempty(d) && continue
    lines!(ax1, d.rho_star, d.F_M; color = col, linestyle = ls, label = meth)
    d4 = cdf[(cdf.ic_type .== "gaussian") .& (cdf.method .== "4LPT") .& (cdf.eps .== 1.0) .& (cdf.growth .== "Dn"), :]; sort!(d4, :rho_star)
    meth != "4LPT" && lines!(ax2, d.rho_star, d.F_M .- d4.F_M; color = col, linestyle = ls, label = "$meth − 4LPT")
    meth == "4LPT" && band!(ax2, d.rho_star, -d.F_M_se, d.F_M_se; color = (:royalblue3, 0.25), label = "4LPT s.e.")
end
d4v = cdf[(cdf.ic_type .== "gaussian") .& (cdf.method .== "4LPT") .& (cdf.eps .== 1.0) .& (cdf.growth .== "Dn"), :]; sort!(d4v, :rho_star)
lines!(ax1, d4v.rho_star, d4v.F_V; color = :royalblue3, linestyle = :dot, label = L"F_V\ \mathrm{(4LPT)}")
# FEM curves: all FEM ensembles present on disk, matched partner, ε = 1
femdirs = filter(d -> startswith(basename(d), "FEM_L500_N") && !occursin("steps", d), readdir(joinpath(M.STUDY_ROOT, "data"); join = true))
fem_cols = Dict(64 => :darkorange, 128 => :firebrick, 256 => :darkred)
for fd in femdirs
    files = filter(f -> startswith(basename(f), "part_fem") && endswith(f, ".jld2") && !occursin("_elements_", f), readdir(fd; join = true))
    isempty(files) && continue
    cdfs = NamedTuple[]; for f in files; append!(cdfs, load(f, "cdf")); end
    N = parse(Int, match(r"_N(\d+)", basename(fd)).captures[1])
    sel(meth, partner) = [c for c in cdfs if c.method == meth && c.eps == 1.0 && c.jtype == "fft" && c.partner == partner]
    fm = sel("FEM", "matched"); l4 = sel("4LPT", "matched")
    isempty(fm) && continue
    FM = mean(hcat((Float64.(c.F_M) for c in fm)...); dims = 2)[:]; F4 = mean(hcat((Float64.(c.F_M) for c in l4)...); dims = 2)[:]
    lines!(ax1, rho_star, FM; color = fem_cols[N], linestyle = :dash, label = "FEM N=$N (L=500, $(length(fm)) seeds)")
    lines!(ax2, rho_star, FM .- F4; color = fem_cols[N], linestyle = :dash, label = "FEM − matched 4LPT, N=$N")
end
hlines!(ax2, [0.0]; color = :black)
axislegend(ax1; position = :rt, framevisible = false, labelsize = 10); axislegend(ax2; position = :rt, framevisible = false, labelsize = 10)
save_fig("fig5_mass_cdf_curves", fig)

# ── Fig 6: convergence of A1, A3 with resolution and box size ────────────────────────────
ens = vcat([main], conv)
fig = Figure(size = (1000, 460))
ax1 = Axis(fig[1, 1]; ylabel = L"A_1", xticks = (1:length(ens), ens), xticklabelrotation = π / 6, title = "4LPT, Gaussian & fixed ICs")
ax2 = Axis(fig[1, 2]; ylabel = L"A_3", xticks = (1:length(ens), ens), xticklabelrotation = π / 6)
for (i, e) in enumerate(ens)
    dir = joinpath(M.STUDY_ROOT, "data", e)
    isdir(dir) || continue
    Ee = try M.load_ensemble(dir) catch; nothing end
    Ee === nothing && continue
    afe = M.amplitude_fits(Ee.rows)
    for ic in sort(unique(afe.summary.ic_type))
        s = afe.summary[(afe.summary.ic_type .== ic) .& (afe.summary.method .== "4LPT") .& (afe.summary.growth .== "Dn") .& (afe.summary.fit .== "O"), :]
        a1 = s[s.power .== 1, :]; a3 = s[s.power .== 3, :]
        off = ic == "gaussian" ? -0.1 : 0.1
        errorbars!(ax1, [i + off], a1.A_mean, a1.A_se; color = colors[ic], whiskerwidth = 8); scatter!(ax1, [i + off], a1.A_mean; color = colors[ic], markersize = 12)
        errorbars!(ax2, [i + off], a3.A_mean, a3.A_se; color = colors[ic], whiskerwidth = 8); scatter!(ax2, [i + off], a3.A_mean; color = colors[ic], markersize = 12)
    end
end
hlines!(ax1, [A1]; color = :black, linestyle = :dash); hlines!(ax2, [A3p]; color = :red, linestyle = :dash); hlines!(ax2, [0.0]; color = :black)
save_fig("fig6_convergence_A1_A3", fig)

# ── Fig 7: shrinking-bin extrapolation of ⟨j₄ | δ_L = 0⟩/σ⁴ ────────────────────────────────
if nrow(E.mcond) > 0
    mx = M.m4_extrapolation(E.mcond)
    fig = Figure(size = (1000, 460))
    ax1 = Axis(fig[1, 1]; xlabel = L"w^2\ \ (|\delta_L|/\sigma < w)", ylabel = L"\langle j_4\rangle_w/\sigma^4", title = "fourth-order conditional coefficient")
    ax2 = Axis(fig[1, 2]; xlabel = L"w^2", ylabel = L"\langle j_2\rangle_w/\sigma^2", title = L"\mathrm{second\ order}\ (\to 4/21)")
    for ic in sort(unique(mx.ic_type))
        d4 = mx[(mx.ic_type .== ic) .& (mx.k .== 4) .& isfinite.(mx.width), :]; d2 = mx[(mx.ic_type .== ic) .& (mx.k .== 2) .& isfinite.(mx.width), :]
        errorbars!(ax1, d4.width .^ 2, d4.mean, d4.se; color = colors[ic]); scatter!(ax1, d4.width .^ 2, d4.mean; color = colors[ic], label = "$ic: w²→0: $(@sprintf("%.4f ± %.4f", d4.extrap_w2[1], d4.extrap_w2_se[1]))")
        xs = range(0, 0.045, length = 10); lines!(ax1, xs, d4.extrap_w2[1] .+ d4.extrap_slope_w2[1] .* xs; color = colors[ic], linestyle = :dot)
        errorbars!(ax2, d2.width .^ 2, d2.mean, d2.se; color = colors[ic]); scatter!(ax2, d2.width .^ 2, d2.mean; color = colors[ic], label = ic)
    end
    hlines!(ax1, [M.M4_PRED]; color = :red, linestyle = :dash, label = "prediction −0.0152"); hlines!(ax2, [4 / 21]; color = :red, linestyle = :dash, label = "4/21")
    axislegend(ax1; position = :rb, framevisible = false, labelsize = 10); axislegend(ax2; position = :rb, framevisible = false, labelsize = 10)
    xlims!(ax1, -0.002, 0.045)
    save_fig("fig7_m4_extrapolation", fig)
end
println("figures in $figdir")
