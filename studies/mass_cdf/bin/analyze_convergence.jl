# Cross-ensemble convergence: resolution (same phases), box size, LPT order, filter scale.
#   julia --project=studies/mass_cdf studies/mass_cdf/bin/analyze_convergence.jl
# Writes results/convergence/{A_coefficients.csv, same_seed_differences.csv, lpt_order.csv}
let py = get(ENV, "DISCODJ_PYTHON", normpath(joinpath(@__DIR__, "..", "..", "..", "..", "disco-dj-fem", ".venv", "bin", "python")))
    get!(ENV, "JULIA_PYTHONCALL_EXE", py); ENV["JULIA_CONDAPKG_BACKEND"] = "Null"; get!(ENV, "JAX_PLATFORMS", "cpu")
end
using MassCDF, DataFrames, Statistics, Printf, JLD2, LinearAlgebra
const M = MassCDF
datadir = joinpath(M.STUDY_ROOT, "data"); resdir = joinpath(M.STUDY_ROOT, "results", "convergence"); mkpath(resdir)
ens = sort(filter(d -> !startswith(d, "FEM_") && !startswith(d, "test") && isdir(joinpath(datadir, d)) &&
                       any(f -> startswith(f, "part_") && endswith(f, ".jld2") && !occursin("_elements_", f), readdir(joinpath(datadir, d))), readdir(datadir)))
println("ensembles: ", ens)
loaded = Dict(e => M.load_ensemble(joinpath(datadir, e)) for e in ens)

# 1. A1, A3 (odd paired fit "O" and line fit) per ensemble, ic type, method
out = Dict{Symbol,Any}[]
for e in ens
    E = loaded[e]; rows = E.rows
    af = M.amplitude_fits(rows); lf = M.yodd_line_fit(af.oddeven)
    for g in groupby(af.summary[(af.summary.fit .== "O") .& (af.summary.growth .== "Dn"), :], [:ic_type, :method])
        a1 = g[g.power .== 1, :]; a3 = g[g.power .== 3, :]; a5 = g[g.power .== 5, :]
        l = lf[(lf.ic_type .== g.ic_type[1]) .& (lf.method .== g.method[1]) .& (lf.growth .== "Dn") .& (lf.fit .== "line"), :]
        u = af.summary[(af.summary.fit .== "U") .& (af.summary.growth .== "Dn") .& (af.summary.ic_type .== g.ic_type[1]) .& (af.summary.method .== g.method[1]), :]
        push!(out, Dict(:ensemble => e, :L => rows.L[1], :N => rows.N[1], :R_f => rows.R_f[1], :ic_type => g.ic_type[1], :method => g.method[1],
            :n_seeds => a1.n_seeds[1], :volume_Gpc3 => a1.n_seeds[1] * (rows.L[1] / 1000)^3, :sigma0 => a1.sigma0[1],
            :A1 => a1.A_mean[1], :A1_se => a1.A_se[1], :A3 => a3.A_mean[1], :A3_se => a3.A_se[1], :A5 => a5.A_mean[1], :A5_se => a5.A_se[1],
            :A1_line => nrow(l) == 2 ? l.value[l.coef .== "A1"][1] : NaN, :A1_line_se => nrow(l) == 2 ? l.se[l.coef .== "A1"][1] : NaN,
            :A3_line => nrow(l) == 2 ? l.value[l.coef .== "A3"][1] : NaN, :A3_line_se => nrow(l) == 2 ? l.se[l.coef .== "A3"][1] : NaN,
            :c2_over_se => nrow(u) > 0 ? u.c_mean[u.power .== 2][1] / u.c_se[u.power .== 2][1] : NaN,
            :c4_over_se => nrow(u) > 0 ? u.c_mean[u.power .== 4][1] / u.c_se[u.power .== 4][1] : NaN,
            :A1_minus_pred => a1.A_mean[1] - M.A1_EDS, :A1_pull => (a1.A_mean[1] - M.A1_EDS) / a1.A_se[1],
            :A3_minus_pred => a3.A_mean[1] - M.A3_PRED, :A3_pull => (a3.A_mean[1] - M.A3_PRED) / a3.A_se[1]))
    end
end
coef = DataFrame(out); sort!(coef, [:R_f, :L, :N, :ensemble, :ic_type, :method])
M.write_csv(joinpath(resdir, "A_coefficients.csv"), coef)

# 2. same-seed differences of Y(ε) between ensembles sharing phases (same L, same N_fine)
function ymat(rows, ic, method, growth)
    d = rows[(rows.ic_type .== ic) .& (rows.method .== method) .& (rows.growth .== growth), :]
    Dict((r.seed, r.eps) => r.Y for r in eachrow(d))
end
pairs_ = [("L1000_N64_R18", "L1000_N128_R18"), ("L1000_N128_R18", "L1000_N256_R18"), ("L1000_N64_R18", "L1000_N256_R18"),
          ("L1000_N128_R18", "L1000_N128_R18_5LPT"), ("L1000_N128_R18_5LPT", "L1000_N128_R18_6LPT")]
sd = Dict{Symbol,Any}[]
for (a, b) in pairs_
    (haskey(loaded, a) && haskey(loaded, b)) || continue
    ra = loaded[a].rows; rb = loaded[b].rows
    for ic in intersect(unique(ra.ic_type), unique(rb.ic_type)), method in ["2LPT", "4LPT"]
        ya = ymat(ra, ic, method, "Dn"); yb = ymat(rb, ic, method, "Dn")
        for ε in sort(unique(abs.(ra.eps)))
            keys_ = [k for k in keys(ya) if abs(k[2]) == ε && haskey(yb, k)]
            isempty(keys_) && continue
            d = [yb[k] - ya[k] for k in keys_]
            e = M.ensemble_errors(d)
            push!(sd, Dict(:ensemble_a => a, :ensemble_b => b, :ic_type => ic, :method => method, :eps => ε, :n_pairs => length(d),
                           :dY_mean => e.mean, :dY_se => e.se, :dY_rms => sqrt(mean(abs2, d)), :Y_a_mean => mean(ya[k] for k in keys_)))
        end
    end
end
sdf = DataFrame(sd); nrow(sdf) > 0 && M.write_csv(joinpath(resdir, "same_seed_differences.csv"), sdf)

# 3. LPT-order convergence of Y(ε=1) and of A1, A3 (from the highest-order ensembles available)
lo = Dict{Symbol,Any}[]
for e in ens
    rows = loaded[e].rows; af = M.amplitude_fits(rows)
    for ic in unique(rows.ic_type)
        meths = sort(unique(rows.method[(rows.growth .== "Dn") .& (rows.ic_type .== ic)]))
        top = meths[argmax([parse(Int, m[1:1]) for m in meths])]
        for m in meths
            s = af.summary[(af.summary.fit .== "O") .& (af.summary.growth .== "Dn") .& (af.summary.ic_type .== ic) .& (af.summary.method .== m), :]
            sub = rows[(rows.growth .== "Dn") .& (rows.ic_type .== ic) .& (rows.eps .== 1.0), :]
            ytop = Dict(r.seed => r.Y for r in eachrow(sub[sub.method .== top, :]))
            ym = sub[sub.method .== m, :]
            d = [r.Y - ytop[r.seed] for r in eachrow(ym)]
            push!(lo, Dict(:ensemble => e, :ic_type => ic, :method => m, :top_method => top, :n_seeds => length(d),
                           :Y_eps1 => mean(ym.Y), :Y_eps1_minus_top => mean(d), :Y_eps1_minus_top_se => M.ensemble_errors(d).se,
                           :A1 => s.A_mean[s.power .== 1][1], :A1_se => s.A_se[s.power .== 1][1], :A3 => s.A_mean[s.power .== 3][1], :A3_se => s.A_se[s.power .== 3][1]))
        end
    end
end
lodf = DataFrame(lo); sort!(lodf, [:ensemble, :ic_type, :method]); M.write_csv(joinpath(resdir, "lpt_order.csv"), lodf)

println("\n=== A1, A3 by ensemble (4LPT, paired odd fit; errors = realization scatter) ===")
for r in eachrow(coef[coef.method .== "4LPT", :])
    @printf("%-22s %-8s N=%3d L=%5.0f R_f=%2.0f n=%3d V=%6.1f Gpc³  A1=%.6f±%.6f (pull %+.1f)  A3=%+.5f±%.5f (line %+.5f±%.5f)  c2/se=%+.1f c4/se=%+.1f\n",
            r.ensemble, r.ic_type, r.N, r.L, r.R_f, r.n_seeds, r.volume_Gpc3, r.A1, r.A1_se, r.A1_pull, r.A3, r.A3_se, r.A3_line, r.A3_line_se, r.c2_over_se, r.c4_over_se)
end
println("\n=== same-seed differences Y_b − Y_a ===")
for r in eachrow(sdf)
    r.eps in (0.5, 1.0) || continue
    @printf("%-20s → %-22s %-8s %-4s ε=%.1f n=%3d  ΔY=%+.2e ± %.2e  rms=%.2e  (Y=%.5f)\n", r.ensemble_a, r.ensemble_b, r.ic_type, r.method, r.eps, r.n_pairs, r.dY_mean, r.dY_se, r.dY_rms, r.Y_a_mean)
end
println("\n=== LPT order convergence ===")
for r in eachrow(lodf)
    @printf("%-22s %-8s %-5s (top %s) n=%3d  Y(ε=1)=%.6f  Y−Y_top=%+.2e ± %.1e   A1=%.6f±%.6f  A3=%+.5f±%.5f\n", r.ensemble, r.ic_type, r.method, r.top_method, r.n_seeds, r.Y_eps1, r.Y_eps1_minus_top, r.Y_eps1_minus_top_se, r.A1, r.A1_se, r.A3, r.A3_se)
end
println("\nwrote $resdir")
