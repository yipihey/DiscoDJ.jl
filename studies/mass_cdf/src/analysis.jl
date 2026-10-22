# ── aggregation and fits (filled in by bin/analyze.jl) ───────────────────────

"Load and concatenate all worker outputs `data/<ensemble>/*.jld2` (excluding element dumps)."
function load_ensemble(dir::AbstractString; prefixes = ("part_gaussian", "part_fixed"))
    files = sort(filter(f -> endswith(f, ".jld2") && !occursin("_elements_", f) && any(p -> startswith(basename(f), p), prefixes),
                        readdir(dir; join = true)))
    isempty(files) && error("no worker outputs in $dir")
    rows = DataFrame[]; cdf = NamedTuple[]; shear = DataFrame[]; mcond = DataFrame[]; metas = Dict[]
    for f in files
        d = load(f)
        push!(rows, d["rows"]); append!(cdf, d["cdf"]); push!(shear, d["shear"])
        haskey(d, "mcond") && push!(mcond, d["mcond"])
        push!(metas, d["meta"])
    end
    return (rows = vcat(rows...; cols = :union), cdf = cdf, shear = vcat(shear...; cols = :union),
            mcond = isempty(mcond) ? DataFrame() : vcat(mcond...; cols = :union), meta = metas)
end

"Write a DataFrame as CSV (no external dependency)."
function write_csv(path::AbstractString, df::DataFrame)
    open(path, "w") do io
        println(io, join(string.(names(df)), ","))
        for r in eachrow(df)
            println(io, join((_csvcell(r[c]) for c in names(df)), ","))
        end
    end
    return path
end
_csvcell(x::AbstractString) = occursin(",", x) ? "\"$x\"" : x
_csvcell(x::AbstractFloat) = isfinite(x) ? @sprintf("%.10g", x) : string(x)
_csvcell(x::Missing) = ""
_csvcell(x) = string(x)

"""
    polyfit_cov(x, y, powers; w=nothing) -> (coef, cov, resid)

Weighted least squares y ≈ Σ c_p x^p over the given `powers`; `w` are per-point weights
(inverse variances).  `cov` is the parameter covariance scaled by the reduced χ² when no
weights are given, and the formal (σ²-based) covariance when weights are given.
"""
function polyfit_cov(x::AbstractVector, y::AbstractVector, powers::AbstractVector{Int}; w = nothing, cov_y = nothing)
    X = hcat((x .^ p for p in powers)...)
    if cov_y !== nothing
        Ci = inv(Symmetric(cov_y))
        A = X' * Ci * X; coef = A \ (X' * Ci * y)
        cov = inv(Symmetric(A))
        return (coef = coef, cov = cov, resid = y - X * coef)
    end
    W = w === nothing ? I : Diagonal(w)
    A = X' * W * X; coef = A \ (X' * W * y)
    resid = y - X * coef
    cov = inv(Symmetric(A))
    if w === nothing
        dof = max(length(y) - length(powers), 1)
        cov *= sum(abs2, resid) / dof
    end
    return (coef = coef, cov = cov, resid = resid)
end

"Mean, standard error (realization scatter), jackknife and bootstrap errors of a per-seed vector."
function ensemble_errors(v::AbstractVector; nboot = 2000, rng = MersenneTwister(1))
    n = length(v); m = mean(v)
    se = n > 1 ? std(v) / sqrt(n) : NaN
    jk = n > 1 ? sqrt((n - 1) / n * sum(abs2, [mean(v[setdiff(1:n, i)]) - m for i in 1:n])) : NaN
    bs = n > 1 ? std([mean(v[rand(rng, 1:n, n)]) for _ in 1:nboot]) : NaN
    return (mean = m, se = se, jackknife = jk, bootstrap = bs, n = n)
end

# ── ensemble-level analysis ───────────────────────────────────────────────────

const A3_PRED = 1.6e-3          # analytic 4th-order prediction near R_f = 18 Mpc/h (to be tested)
const M4_PRED = -0.0152         # analytic ⟨j₄ | δ_L = 0⟩/σ⁴ near R_f = 18 Mpc/h (to be tested)

"Group keys identifying a (method, growth, epoch) family."
const FAMILY = [:ic_type, :method, :growth, :a]

"Mean and standard error of a column within groups."
function group_mean_se(df::DataFrame, keys::Vector{Symbol}, cols::Vector{Symbol})
    g = groupby(df, keys)
    out = combine(g, nrow => :n_seeds,
                  [c => (x -> mean(skipmissing(x))) => Symbol(c, "_mean") for c in cols]...,
                  [c => (x -> (n = count(!ismissing, x); n > 1 ? std(skipmissing(x)) / sqrt(n) : NaN)) => Symbol(c, "_se") for c in cols]...)
    return out
end

"""
    amplitude_fits(rows) -> (per_seed, summary, cov, oddeven)

Paired amplitude expansion.  With Y ≡ F_M(>ρ̄) − P(δ_L > 0), the sign flip δ₀ → −δ₀ leaves
F_M − 1/2 = A₁σ + A₃σ³ + … unchanged (σ = |ε|σ₀) while P(δ_L>0) − 1/2 changes sign, so

    Y_even(ε) = [Y(+ε) + Y(−ε)]/2 = [F_M(+ε) + F_M(−ε)]/2 − 1/2      (physical signal, zero-threshold noise cancels exactly)
    Y_odd(ε)  = [Y(+ε) − Y(−ε)]/2 = [F_M(+ε) − F_M(−ε)]/2 + 1/2 − P(δ₀>0)   (pure noise, zero in expectation).

The amplitude expansion is therefore fitted to Y_pair ≡ Y_even over |ε| ∈ {0.2 … 1}:
  (U)  Y_pair = c₁ε + c₂ε² + c₃ε³ + c₄ε⁴ + c₅ε⁵   (tests c₂ = c₄ = 0),
  (O)  Y_pair = c₁ε + c₃ε³ + c₅ε⁵,
  (O3) Y_pair = c₁ε + c₃ε³ on |ε| ≤ 0.65,
  (S)  single-sign Y(+ε) = c₀ + c₁ε + c₃ε³ + c₅ε⁵ (c₀ absorbs the finite-box offset; cross-check).
Per-seed coefficients, ensemble means with realization-scatter / jackknife / bootstrap errors,
GLS fits to the ensemble-mean curve with the empirical ε–ε covariance, and A_k = c_k/σ₀ᵏ with σ₀
the ensemble-mean σ at ε = 1.
"""
function amplitude_fits(rows::DataFrame)
    per_seed = DataFrame[]; summary = Dict{Symbol,Any}[]; covs = Dict{String,Any}(); oddeven = DataFrame[]
    for fam in groupby(rows, FAMILY)
        key = (ic_type = fam.ic_type[1], method = fam.method[1], growth = fam.growth[1], a = fam.a[1])
        seeds = sort(unique(fam.seed))
        εs = sort(unique(abs.(fam.eps))); nε = length(εs)
        σ0 = mean(fam.sigma[fam.eps .== 1.0])
        σ0_box = fam.sigma_box[findfirst(==(1.0), fam.eps)]
        Yp = fill(NaN, length(seeds), nε); Ym = fill(NaN, length(seeds), nε)
        for (i, s) in enumerate(seeds)
            sub = fam[fam.seed .== s, :]
            for r in eachrow(sub)
                j = findfirst(==(abs(r.eps)), εs)
                r.eps > 0 ? (Yp[i, j] = r.Y) : (Ym[i, j] = r.Y)
            end
        end
        haspair = all(isfinite, Ym)
        Ypair = haspair ? (Yp .+ Ym) ./ 2 : Yp
        Yodd = haspair ? (Yp .- Ym) ./ 2 : fill(NaN, size(Yp))
        fits = Dict("U" => (powers = [1, 2, 3, 4, 5], mask = trues(nε), Y = Ypair),
                    "O" => (powers = [1, 3, 5], mask = trues(nε), Y = Ypair),
                    "O3" => (powers = [1, 3], mask = εs .<= 0.65 .+ 1e-9, Y = Ypair),
                    "S" => (powers = [0, 1, 3, 5], mask = trues(nε), Y = Yp))
        ps = DataFrame(seed = seeds)
        for (name, f) in fits
            C = fill(NaN, length(seeds), length(f.powers))
            count(f.mask) > length(f.powers) || continue
            for i in eachindex(seeds)
                C[i, :] = polyfit_cov(εs[f.mask], f.Y[i, f.mask], f.powers).coef
            end
            for (j, p) in enumerate(f.powers)
                ps[!, Symbol("c$(p)_$name")] = C[:, j]
            end
            n = length(seeds)
            m = vec(mean(C; dims = 1)); cov_mean = n > 1 ? cov(C) / n : fill(NaN, size(C, 2), size(C, 2))
            Ybar = vec(mean(f.Y[:, f.mask]; dims = 1))
            covY = n > 1 ? cov(f.Y[:, f.mask]) / n : nothing
            gls = n > length(f.powers) + 2 ? polyfit_cov(εs[f.mask], Ybar, f.powers; cov_y = covY + 1e-30I) : nothing
            for (j, p) in enumerate(f.powers)
                d = Dict{Symbol,Any}(:ic_type => key.ic_type, :method => key.method, :growth => key.growth, :a => key.a,
                    :fit => name, :power => p, :n_seeds => n, :sigma0 => σ0, :sigma0_box => σ0_box,
                    :c_mean => m[j], :c_se => sqrt(cov_mean[j, j]),
                    :c_jackknife => ensemble_errors(C[:, j]).jackknife, :c_bootstrap => ensemble_errors(C[:, j]).bootstrap,
                    :A_mean => m[j] / σ0^p, :A_se => sqrt(cov_mean[j, j]) / σ0^p,
                    :c_gls => gls === nothing ? NaN : gls.coef[j], :c_gls_se => gls === nothing ? NaN : sqrt(gls.cov[j, j]),
                    :A_gls => gls === nothing ? NaN : gls.coef[j] / σ0^p, :A_gls_se => gls === nothing ? NaN : sqrt(gls.cov[j, j]) / σ0^p,
                    :chi2_gls => gls === nothing ? NaN : (gls.resid' * inv(Symmetric(covY + 1e-30I)) * gls.resid), :dof_gls => gls === nothing ? 0 : count(f.mask) - length(f.powers))
                push!(summary, d)
            end
            covs["$(key.ic_type)/$(key.method)/$(key.growth)/a$(key.a)/$name"] = Dict("powers" => f.powers, "cov_of_mean" => cov_mean,
                "cov_realization" => n > 1 ? cov(C) : nothing, "gls_cov" => gls === nothing ? nothing : gls.cov, "sigma0" => σ0)
        end
        for c in FAMILY
            ps[!, c] .= getfield(key, c)
        end
        push!(per_seed, ps)
        for (j, ε) in enumerate(εs)
            σε = mean(fam.sigma[abs.(fam.eps) .== ε])
            ep = ensemble_errors(Ypair[:, j]); eo = haspair ? ensemble_errors(Yodd[:, j]) : (mean = NaN, se = NaN, jackknife = NaN, bootstrap = NaN)
            push!(oddeven, DataFrame(ic_type = key.ic_type, method = key.method, growth = key.growth, a = key.a, eps = ε, sigma = σε, sigma2 = σε^2,
                n_seeds = length(seeds), Y_pair = ep.mean, Y_pair_se = ep.se, Y_pair_jk = ep.jackknife, Y_pair_bs = ep.bootstrap,
                Y_even = ep.mean, Y_even_se = ep.se,
                Y_odd = eo.mean, Y_odd_se = eo.se, Y_odd_jk = eo.jackknife, Y_odd_bs = eo.bootstrap,
                Y_plus = mean(Yp[:, j]), Y_plus_se = ensemble_errors(Yp[:, j]).se, Y_minus = haspair ? mean(Ym[:, j]) : NaN,
                Y_pair_over_sigma = ep.mean / σε, Y_pair_over_sigma_se = ep.se / σε,
                A1_pred_term = A1_EDS * σε, A3_pred_term = A3_PRED * σε^3))
        end
    end
    return (per_seed = vcat(per_seed...; cols = :union), summary = DataFrame(summary), cov = covs, oddeven = vcat(oddeven...; cols = :union))
end

"""
    yodd_line_fit(oddeven) -> DataFrame

Fit Y_pair/σ = A₁ + A₃σ² (+ A₅σ⁴) across the amplitude grid with the realization-scatter
errors (the straight-line representation of Fig. 1).
"""
function yodd_line_fit(oe::DataFrame)
    out = Dict{Symbol,Any}[]
    for fam in groupby(oe, FAMILY)
        x = fam.sigma2; y = fam.Y_pair_over_sigma; w = 1 ./ fam.Y_pair_over_sigma_se .^ 2
        for (name, powers) in (("line", [0, 1]), ("quad", [0, 1, 2]))
            length(x) <= length(powers) && continue
            f = polyfit_cov(x, y, powers; w = w)
            chi2 = sum(w .* f.resid .^ 2)
            for (j, p) in enumerate(powers)
                push!(out, Dict(:ic_type => fam.ic_type[1], :method => fam.method[1], :growth => fam.growth[1], :a => fam.a[1],
                    :fit => name, :coef => "A$(2p+1)", :value => f.coef[j], :se => sqrt(f.cov[j, j]), :chi2 => chi2, :dof => length(x) - length(powers)))
            end
        end
    end
    return DataFrame(out)
end

"""
    m4_extrapolation(mcond) -> DataFrame

Ensemble mean (± se over seeds) of ⟨jₖ⟩/σᵏ in the symmetric shrinking bins |δ_L|/σ < w, and the
w → 0 extrapolation: weighted linear fit in w² over bins with w ≤ 0.2 (the conditional mean is an
even function of δ_L at leading order, so the bin average approaches the point value as w²).
"""
function m4_extrapolation(mc::DataFrame)
    out = Dict{Symbol,Any}[]
    hasvar = "variant" in names(mc)
    for g in groupby(mc, hasvar ? [:ic_type, :variant, :k] : [:ic_type, :k])
        tab = combine(groupby(g, :width), :mean => mean => :m, :mean => (x -> std(x) / sqrt(length(x))) => :se,
                      :n => sum => :n_total, nrow => :n_seeds, :std => mean => :std_within)
        sort!(tab, :width)
        qi = combine(groupby(g, :seed), :quad_intercept => first => :q).q
        fin = tab[isfinite.(tab.width) .& (tab.width .<= 0.2 .+ 1e-9), :]
        f = length(fin.width) >= 3 ? polyfit_cov(fin.width .^ 2, fin.m, [0, 1]; w = 1 ./ fin.se .^ 2) : nothing
        for r in eachrow(tab)
            push!(out, Dict(:ic_type => g.ic_type[1], :variant => hasvar ? g.variant[1] : "lnJ", :k => g.k[1], :width => r.width, :mean => r.m, :se => r.se, :std_within => r.std_within,
                :n_total => r.n_total, :n_seeds => r.n_seeds,
                :extrap_w2 => f === nothing ? NaN : f.coef[1], :extrap_w2_se => f === nothing ? NaN : sqrt(f.cov[1, 1]),
                :extrap_slope_w2 => f === nothing ? NaN : f.coef[2],
                :quad_intercept_mean => mean(qi), :quad_intercept_se => std(qi) / sqrt(length(qi))))
        end
    end
    return DataFrame(out)
end

"Long-format ensemble-mean CDF curves (F_M, F_V with standard errors) per family and ε."
function cdf_curves_table(cdf::Vector{<:NamedTuple}, rho_star::Vector{Float64})
    out = Dict{Symbol,Any}[]
    groups = Dict{Any,Vector{Int}}()
    for (i, c) in enumerate(cdf)
        push!(get!(groups, (c.ic_type, c.method, c.growth, c.a, c.eps), Int[]), i)
    end
    for (k, idx) in groups
        FM = hcat((Float64.(cdf[i].F_M) for i in idx)...); FV = hcat((Float64.(cdf[i].F_V) for i in idx)...)
        n = length(idx)
        for (j, ρ) in enumerate(rho_star)
            push!(out, Dict(:ic_type => k[1], :method => k[2], :growth => k[3], :a => k[4], :eps => k[5], :rho_star => ρ, :n_seeds => n,
                :F_M => mean(FM[j, :]), :F_M_se => n > 1 ? std(FM[j, :]) / sqrt(n) : NaN,
                :F_V => mean(FV[j, :]), :F_V_se => n > 1 ? std(FV[j, :]) / sqrt(n) : NaN))
        end
    end
    df = DataFrame(out); sort!(df, [:ic_type, :method, :growth, :a, :eps, :rho_star]); return df
end
