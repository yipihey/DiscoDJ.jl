include(joinpath(@__DIR__, "_env.jl"))
include(joinpath(@__DIR__, "_csv.jl"))
using CairoMakie, DataFrames
set_theme!(theme_latexfonts())
resdir = joinpath(M.STUDY_ROOT, "results", "multiscale")
sc = readcsv(joinpath(resdir, "scale_correlations.csv"))
loc = readcsv(joinpath(resdir, "locality_gain.csv"))
cols = Makie.wong_colors()
# ---- fig 1: f_new vs R_f
fig = Figure(size = (1100, 420))
for (j, (s, ttl)) in enumerate((("0", "trace sector"), ("E", "diagonal traceless (E)"), ("T", "off-diagonal (T)")))
    ax = Axis(fig[1, j]; xlabel = "R_f [Mpc/h]", ylabel = j == 1 ? "1 - r_s^2" : "", yscale = log10, title = ttl, xticks = [14, 18, 22])
    for (ci, N) in enumerate((64, 128, 256))
        b = sc[(sc.method .== "box") .& (sc.cosmo .== "lcdm") .& (sc.N .== N), :]; sort!(b, :R_f)
        lines!(ax, b.R_f, b[:, "f$(s)_new"]; color = cols[ci], label = "box sum, h=$(round(1000 / N; digits = 1))")
        ct = sc[(sc.method .== "continuum") .& (sc.cosmo .== "lcdm") .& (sc.N .== N), :]; sort!(ct, :R_f)
        scatter!(ax, ct.R_f, ct[:, "f$(s)_new"]; color = cols[ci], marker = :xcross, markersize = 11)
        m = sc[(sc.method .== "measured") .& (sc.cosmo .== "lcdm") .& (sc.N .== N), :]; sort!(m, :R_f)
        isempty(m) || scatter!(ax, m.R_f, m[:, "f$(s)_meas"]; color = cols[ci], marker = :circle, markersize = 9, strokewidth = 1, strokecolor = :black)
    end
    j == 1 && axislegend(ax; position = :lt, labelsize = 11, framevisible = false)
end
Label(fig[0, :], "line: box-sum prediction;  x: continuum integral;  circle: measured (seeds 1-8, 1-4 at N=256, L=1000)"; fontsize = 12)
save(joinpath(resdir, "fig_fnew_vs_Rf.png"), fig; px_per_unit = 2)
# ---- fig 2: residual rms one vs two (dumbbell), invariants, ln J, cut J>0.1
l = loc[(loc.model .== "invariants") .& (loc.target .== "lnJ") .& (loc.cut .== "J>0.1"), :]
function lab(r)
    geo = r.L == 1000 ? "L1000 N128 R$(Int(r.R_f))" : "L500 N$(r.N)"
    part = (hasproperty(r, :partner) && r.partner isa AbstractString && r.partner != "-") ? " " * r.partner : ""
    src = r.A_source == "input" ? "A_in" : r.A_source == "fem_lin" ? "A_femlin" : "A_K"
    return "$(r.method)$part $geo eps=$(r.eps) [$src]"
end
l.label = [lab(r) for r in eachrow(l)]
sort!(l, [:L, :N, :method, :R_f, :A_source, :eps])
fig = Figure(size = (1000, 14 * nrow(l) + 120))
ax = Axis(fig[1, 1]; xlabel = "rms residual of ln J (cubes with J > 0.1, test set)", xscale = log10, yticks = (1:nrow(l), l.label), yticklabelsize = 9, yreversed = true)
for i in 1:nrow(l)
    lines!(ax, [l.rms_res_one[i], l.rms_res_two[i]], [i, i]; color = :gray70)
end
scatter!(ax, l.rms_res_one, 1:nrow(l); color = cols[1], markersize = 8, label = "one-scale f(A_C)")
scatter!(ax, l.rms_res_two, 1:nrow(l); color = cols[2], markersize = 8, label = "two-scale g(A_C, A_P)")
axislegend(ax; position = :rb)
save(joinpath(resdir, "fig_residual_rms.png"), fig; px_per_unit = 2)
# ---- fig 3: R_gain vs R_f
fig = Figure(size = (900, 460))
ax = Axis(fig[1, 1]; xlabel = "R_f [Mpc/h]", ylabel = "R_gain = 1 - V_CP / V_C", xticks = [14, 18, 22], title = "ln J, cut J > 0.1, eps = 1 (L = 1000, N = 128, h = 7.8)")
for (k, (meth, src, mod, lab, col, mk)) in enumerate((("4LPT", "input", "invariants", "4LPT, invariants", cols[1], :circle), ("2LPT", "input", "invariants", "2LPT, invariants", cols[2], :rect),
                                                        ("4LPT", "input", "flexible", "4LPT, flexible (deg 3)", cols[3], :utriangle), ("4LPT", "A_K(Kuhn edge tensor)", "invariants", "4LPT, A_K instead of A_C", cols[4], :diamond),
                                                        ("2LPT", "A_K(Kuhn edge tensor)", "invariants", "2LPT, A_K instead of A_C", cols[5], :dtriangle)))
    b = loc[(loc.L .== 1000) .& (loc.method .== meth) .& (loc.A_source .== src) .& (loc.model .== mod) .& (loc.target .== "lnJ") .& (loc.cut .== "J>0.1") .& (loc.eps .== 1.0), :]
    isempty(b) && continue
    sort!(b, :R_f)
    errorbars!(ax, b.R_f, b.R_gain, coalesce.(b.R_gain_err, 0.0); color = col)
    scatterlines!(ax, b.R_f, b.R_gain; color = col, marker = mk, label = lab)
end
lq = joinpath(resdir, "diag_quadrature.csv")
if isfile(lq)
    dq = readcsv(lq); dq = dq[.!ismissing.(dq.R_gain_linear_level), :]
    scatter!(ax, dq.R_f, dq.R_gain_linear_level; color = :black, marker = :hline, markersize = 18, label = "linear-level Gaussian prediction")
end
ylims!(ax, -0.05, 1.45)
axislegend(ax; position = :rt, labelsize = 11, nbanks = 2)
save(joinpath(resdir, "fig_Rgain_vs_Rf.png"), fig; px_per_unit = 2)
# ---- fig 4: innovation response
inf = joinpath(resdir, "innovation_response.csv")
if isfile(inf)
    inn = readcsv(inf); inn = inn[(inn.eps .== 1.0) .& (inn.method .== "4LPT") .& (inn.L .== 1000), :]
    fig = Figure(size = (1000, 400))
    for (j, (nm, ttl)) in enumerate((("eta0abs", "|eta_0| (trace)"), ("EE", "eta_E : eta_E"), ("TT", "eta_T : eta_T")))
        ax = Axis(fig[1, j]; xlabel = "tertile of eps tr A_P", ylabel = j == 1 ? "corr(ln J - f(A_C), x)" : "", title = ttl, xticks = 1:3)
        for (ci, R) in enumerate((14.0, 18.0, 22.0))
            b = inn[(inn.R_f .== R) .& (inn.tertile .> 0), :]; sort!(b, :tertile)
            scatterlines!(ax, b.tertile, b[:, "corr_$nm"]; color = cols[ci], label = "R_f=$(Int(R))")
        end
        hlines!(ax, [0.0]; color = :gray, linestyle = :dash)
        j == 1 && axislegend(ax; position = :lt, framevisible = false)
    end
    save(joinpath(resdir, "fig_innovation.png"), fig; px_per_unit = 2)
end
println("figures written")
