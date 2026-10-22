# Aggregate the FEM-vs-LPT ensembles (data/FEM_*) into results/fem/*.csv
#   julia --project=studies/mass_cdf studies/mass_cdf/bin/analyze_fem.jl [FEM_L500_N64 FEM_L500_N128 ...]
let py = get(ENV, "DISCODJ_PYTHON", normpath(joinpath(@__DIR__, "..", "..", "..", "..", "disco-dj-fem", ".venv", "bin", "python")))
    get!(ENV, "JULIA_PYTHONCALL_EXE", py); ENV["JULIA_CONDAPKG_BACKEND"] = "Null"; get!(ENV, "JAX_PLATFORMS", "cpu")
end
using MassCDF, DataFrames, Statistics, Printf, JLD2, LinearAlgebra
const M = MassCDF
datadir = joinpath(M.STUDY_ROOT, "data")
names_ = isempty(ARGS) ? sort(filter(d -> startswith(d, "FEM_"), readdir(datadir))) : ARGS
resdir = joinpath(M.STUDY_ROOT, "results", "fem"); mkpath(resdir)
rows = DataFrame[]; cdfs = NamedTuple[]; lins = DataFrame[]; metas = Dict[]
for nm in names_
    for f in filter(f -> startswith(basename(f), "part_fem") && endswith(f, ".jld2") && !occursin("_elements_", f), readdir(joinpath(datadir, nm); join = true))
        d = load(f)
        r = d["rows"]; r.ensemble .= nm
        push!(rows, r); append!(cdfs, [(c..., ensemble = nm) for c in d["cdf"]])
        l = d["linear"]; l.ensemble .= nm; push!(lins, l); push!(metas, d["meta"])
    end
end
isempty(rows) && error("no FEM outputs found")
rows = vcat(rows...; cols = :union); lins = vcat(lins...; cols = :union)
println("FEM rows: $(nrow(rows)) from $(names_)")
M.write_csv(joinpath(resdir, "fem_realization_summary.csv"), rows)
M.write_csv(joinpath(resdir, "fem_linear_response.csv"), lins)

# per (ensemble, ε, jtype, partner): Y of FEM and of each LPT order, and the paired differences on the same seeds
out = Dict{Symbol,Any}[]
for g in groupby(rows, [:ensemble, :eps, :jtype, :partner])
    seeds = sort(unique(g.seed))
    Y(meth) = [g.Y[(g.seed .== s) .& (g.method .== meth)][1] for s in seeds]
    "FEM" in g.method || continue
    yf = Y("FEM")
    for meth in ["1LPT", "2LPT", "3LPT", "4LPT"]
        meth in g.method || continue
        yl = Y(meth); d = yf .- yl
        e = M.ensemble_errors(d)
        push!(out, Dict(:ensemble => g.ensemble[1], :N => g.N[1], :h => g.h[1], :eps => g.eps[1], :jtype => g.jtype[1], :partner => g.partner[1],
                        :lpt => meth, :n_seeds => length(seeds), :sigma => mean(g.sigma), :Y_FEM => mean(yf), :Y_FEM_se => M.ensemble_errors(yf).se,
                        :Y_LPT => mean(yl), :Y_LPT_se => M.ensemble_errors(yl).se, :dY_mean => e.mean, :dY_se => e.se, :dY_jackknife => e.jackknife,
                        :dY_rms_per_seed => sqrt(mean(abs2, d)), :dY_over_Y => e.mean / mean(yl),
                        :rel_rms_psi_fem_minus_4lpt => mean(g.rel_rms_fem_minus_4lpt), :minJ_fem_min => minimum(g.minJ[g.method .== "FEM"]),
                        :frac_Jle0_fem_max => maximum(g.frac_Jle0[g.method .== "FEM"])))
    end
end
comp = DataFrame(out); sort!(comp, [:ensemble, :partner, :jtype, :lpt, :eps])
M.write_csv(joinpath(resdir, "fem_vs_lpt.csv"), comp)

# odd/even decomposition of Y_FEM − Y_4LPT (matched and raw)
oe = Dict{Symbol,Any}[]
for g in groupby(rows[(rows.jtype .== "fft"), :], [:ensemble, :partner])
    seeds = sort(unique(g.seed))
    for ε in sort(unique(abs.(g.eps)))
        sel(m, e) = [g.Y[(g.seed .== s) .& (g.method .== m) .& (g.eps .== e)][1] for s in seeds]
        any((g.method .== "FEM") .& (g.eps .== -ε)) || continue
        df = sel("FEM", ε) .- sel("4LPT", ε); dm = sel("FEM", -ε) .- sel("4LPT", -ε)
        odd = (df .- dm) ./ 2; even = (df .+ dm) ./ 2
        yfo = (sel("FEM", ε) .- sel("FEM", -ε)) ./ 2; y4o = (sel("4LPT", ε) .- sel("4LPT", -ε)) ./ 2
        push!(oe, Dict(:ensemble => g.ensemble[1], :partner => g.partner[1], :eps => ε, :n_seeds => length(seeds),
                       :dY_odd => mean(odd), :dY_odd_se => M.ensemble_errors(odd).se, :dY_even => mean(even), :dY_even_se => M.ensemble_errors(even).se,
                       :Y_FEM_odd => mean(yfo), :Y_FEM_odd_se => M.ensemble_errors(yfo).se, :Y_4LPT_odd => mean(y4o), :Y_4LPT_odd_se => M.ensemble_errors(y4o).se,
                       ))
    end
end
oedf = DataFrame(oe); sort!(oedf, [:ensemble, :partner, :eps])
M.write_csv(joinpath(resdir, "fem_odd_even.csv"), oedf)

# CDF differences FEM − 4LPT (matched, fft) per ensemble and ε, with seed scatter
rho_star = metas[1]["rho_star"]
cd = Dict{Symbol,Any}[]
for nm in names_, ε in sort(unique(rows.eps)), partner in ("raw", "matched")
    fm = [c for c in cdfs if c.ensemble == nm && c.method == "FEM" && c.eps == ε && c.jtype == "fft" && c.partner == partner]
    isempty(fm) && continue
    seeds = sort(unique(c.seed for c in fm))
    pick(meth) = hcat((Float64.([c for c in cdfs if c.ensemble == nm && c.method == meth && c.eps == ε && c.jtype == "fft" && c.partner == partner && c.seed == s][1].F_M) for s in seeds)...)
    FM = pick("FEM"); F4 = pick("4LPT"); F2 = pick("2LPT")
    for (j, ρ) in enumerate(rho_star)
        d4 = FM[j, :] .- F4[j, :]; d2 = F2[j, :] .- F4[j, :]
        push!(cd, Dict(:ensemble => nm, :eps => ε, :partner => partner, :rho_star => ρ, :n_seeds => length(seeds),
                       :F_M_FEM => mean(FM[j, :]), :F_M_4LPT => mean(F4[j, :]), :F_M_2LPT => mean(F2[j, :]),
                       :dFM_FEM_minus_4LPT => mean(d4), :dFM_FEM_minus_4LPT_se => length(seeds) > 1 ? std(d4) / sqrt(length(seeds)) : NaN,
                       :dFM_2LPT_minus_4LPT => mean(d2), :dFM_2LPT_minus_4LPT_se => length(seeds) > 1 ? std(d2) / sqrt(length(seeds)) : NaN))
    end
end
M.write_csv(joinpath(resdir, "fem_cdf_differences.csv"), DataFrame(cd))

# Richardson-style mesh extrapolation of the matched ΔY on seeds common to all ensembles (h² scaling assumed)
rich = Dict{Symbol,Any}[]
m = comp[(comp.partner .== "matched") .& (comp.jtype .== "fft") .& (comp.lpt .== "4LPT"), :]
for ε in sort(unique(m.eps))
    d = m[m.eps .== ε, :]; sort!(d, :h; rev = true)
    nrow(d) >= 2 || continue
    h = d.h; y = d.dY_mean
    # fit ΔY(h) = a + b h² over available meshes (weighted by se)
    w = 1 ./ max.(d.dY_se, 1e-7) .^ 2
    f = M.polyfit_cov(h .^ 2, y, [0, 1]; w = w)
    push!(rich, Dict(:eps => ε, :meshes => join(string.(Int.(round.(d.N))), "/"), :dY_h0 => f.coef[1], :dY_h0_se => sqrt(f.cov[1, 1]), :slope_h2 => f.coef[2],
                     :dY_finest => y[end], :dY_finest_se => d.dY_se[end], :Y_4LPT_finest => d.Y_LPT[end]))
end
M.write_csv(joinpath(resdir, "fem_richardson.csv"), DataFrame(rich))

println("\n=== FEM − nLPT, Y at ρ★ = ρ̄ (fft Jacobian) ===")
for r in eachrow(comp[(comp.jtype .== "fft") .& (comp.lpt .∈ Ref(["2LPT", "4LPT"])), :])
    @printf("%-14s h=%5.2f ε=%+.2f %-7s vs %-4s  n=%2d  Y_FEM=%+.5f  Y_LPT=%+.5f  ΔY=%+.5f ± %.5f  (ΔY/Y=%+.3f)  rms(ψ)=%.4f  minJ=%.3f\n",
            r.ensemble, r.h, r.eps, r.partner, r.lpt, r.n_seeds, r.Y_FEM, r.Y_LPT, r.dY_mean, r.dY_se, r.dY_over_Y, r.rel_rms_psi_fem_minus_4lpt, r.minJ_fem_min)
end
println("\n=== tet-volume Jacobian (Lagrangian elements), raw partner, 4LPT ===")
for r in eachrow(comp[(comp.jtype .== "tet") .& (comp.lpt .== "4LPT"), :])
    @printf("%-14s h=%5.2f ε=%+.2f  Y_FEM,tet=%+.5f  Y_4LPT,tet=%+.5f  ΔY=%+.5f ± %.5f\n", r.ensemble, r.h, r.eps, r.Y_FEM, r.Y_LPT, r.dY_mean, r.dY_se)
end
println("\n=== linear response of the FEM operator ===")
for g in groupby(lins, :ensemble)
    @printf("%-14s h=%5.2f  std(δ_FEM)/std(δ)=%.4f  corr(δ,δ_FEM)=%.5f  rms(ψ_lin)/rms(ψ1)=%.4f  rms(ψ_lin−ψ1)/rms(ψ1)=%.4f  ψ2,FEM/ψ2=%.4f (corr %.4f)  n=%d\n",
            g.ensemble[1], g.h[1], mean(g.sigma_fem_lin ./ g.sigma0), mean(g.corr_delta), mean(g.rms_psi_lin_over_psi1), mean(g.rel_rms_psi_lin_minus_psi1),
            mean(g.rms_psi2fem_over_psi2), mean(g.corr_psi2fem_psi2), nrow(g))
end
println("\n=== odd/even parts of Y_FEM − Y_4LPT ===")
for r in eachrow(oedf)
    @printf("%-14s %-7s ε=%.2f n=%d  ΔY_odd=%+.5f ± %.5f  ΔY_even=%+.5f ± %.5f\n", r.ensemble, r.partner, r.eps, r.n_seeds, r.dY_odd, r.dY_odd_se, r.dY_even, r.dY_even_se)
end
println("\n=== mesh extrapolation of matched ΔY (a + b h²) ===")
for r in eachrow(DataFrame(rich))
    @printf("ε=%+.2f meshes N=%s  ΔY(h→0)=%+.5f ± %.5f  finest ΔY=%+.5f ± %.5f  Y_4LPT=%+.5f\n", r.eps, r.meshes, r.dY_h0, r.dY_h0_se, r.dY_finest, r.dY_finest_se, r.Y_4LPT_finest)
end
println("\nwrote $resdir")
