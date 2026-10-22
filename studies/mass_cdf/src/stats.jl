# ── per-realization statistics of the Jacobian field ─────────────────────────

const SM3 = SMatrix{3,3,Float32,9}
const SM3d = SMatrix{3,3,Float64,9}

"Threshold grid ρ★/ρ̄: logarithmic 10^(-1 … 1.3) in steps of 0.05, plus the required points."
function default_rho_star()
    g = 10.0 .^ collect(-1.0:0.05:1.3)
    req = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]
    return sort(unique(round.(vcat(g, req); digits = 10)))
end

Base.@kwdef struct StatsConfig
    rho_star::Vector{Float64} = default_rho_star()
    Jstar::Vector{Float64} = sort(1.0 ./ default_rho_star())
    shear_widths::Vector{Float64} = [0.02, 0.05, 0.1, 0.2]
    hist_nbins::Int = 6000
    hist_Jmax::Float64 = 3.0
    pct::Vector{Float64} = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1]
end
StatsConfig(rho_star::Vector{Float64}; kw...) = StatsConfig(; rho_star, Jstar = sort(1.0 ./ rho_star), kw...)

mutable struct Accum
    n::Int; sumJ::Float64; sumJinv::Float64; nJinv::Int; nJle0::Int; minJ::Float64; maxJ::Float64
    minsv::Float64
    ndpos::Int; n_pp::Int; n_pm::Int; n_mp::Int; n_mm::Int; nJlt1::Int
    hist::Vector{Int}; hunder::Int; hover::Int
    cdf_cnt::Vector{Int}; cdf_sumJ::Vector{Float64}
    sh_n::Vector{Int}; sh_x::Vector{Float64}; sh_y::Vector{Float64}; sh_xx::Vector{Float64}; sh_xy::Vector{Float64}; sh_yy::Vector{Float64}; sh_excl::Vector{Int}
    sum_lnJ::Float64; n_lnJ::Int
end
Accum(cfg::StatsConfig) = Accum(0, 0.0, 0.0, 0, 0, Inf, -Inf, Inf, 0, 0, 0, 0, 0, 0,
    zeros(Int, cfg.hist_nbins), 0, 0, zeros(Int, length(cfg.Jstar) + 1), zeros(Float64, length(cfg.Jstar) + 1),
    zeros(Int, length(cfg.shear_widths)), zeros(length(cfg.shear_widths)), zeros(length(cfg.shear_widths)),
    zeros(length(cfg.shear_widths)), zeros(length(cfg.shear_widths)), zeros(length(cfg.shear_widths)),
    zeros(Int, length(cfg.shear_widths)), 0.0, 0)

function merge_accum!(a::Accum, b::Accum)
    a.n += b.n; a.sumJ += b.sumJ; a.sumJinv += b.sumJinv; a.nJinv += b.nJinv; a.nJle0 += b.nJle0
    a.minJ = min(a.minJ, b.minJ); a.maxJ = max(a.maxJ, b.maxJ); a.minsv = min(a.minsv, b.minsv)
    a.ndpos += b.ndpos; a.n_pp += b.n_pp; a.n_pm += b.n_pm; a.n_mp += b.n_mp; a.n_mm += b.n_mm; a.nJlt1 += b.nJlt1
    a.hist .+= b.hist; a.hunder += b.hunder; a.hover += b.hover
    a.cdf_cnt .+= b.cdf_cnt; a.cdf_sumJ .+= b.cdf_sumJ
    a.sh_n .+= b.sh_n; a.sh_x .+= b.sh_x; a.sh_y .+= b.sh_y; a.sh_xx .+= b.sh_xx; a.sh_xy .+= b.sh_xy; a.sh_yy .+= b.sh_yy; a.sh_excl .+= b.sh_excl
    a.sum_lnJ += b.sum_lnJ; a.n_lnJ += b.n_lnJ
    return a
end

@inline function min_singular_value(F::SM3d)
    C = Symmetric(F' * F)
    λ = eigvals(C)
    return sqrt(max(minimum(λ), 0.0))
end

"""
    jacobian_stats(terms, δL, TT, σ, cfg) -> NamedTuple

`terms` is a vector of `(coefficient, G::Vector{SM3})` pairs; F = I + Σ cᵢ Gᵢ per element.
`δL` is the signed linear density at the epoch/amplitude under test, `TT` the matching T:T
(both already scaled by ε), `σ` the realization rms of δL.  Returns all normalization,
shell-crossing, CDF, transition and shear-regression statistics.
"""
function jacobian_stats(terms::Vector{Tuple{Float64,Vector{SM3}}}, δL::Vector{Float32}, TT::Vector{Float32},
                        σ::Float64, cfg::StatsConfig)
    n = length(δL)
    for (_, G) in terms
        @assert length(G) == n
    end
    nt = Threads.nthreads()
    accs = [Accum(cfg) for _ in 1:nt]
    chunks = Iterators.partition(1:n, cld(n, nt))
    I3 = SM3d(I)
    Jstar = cfg.Jstar; nJs = length(Jstar); widths = cfg.shear_widths .* σ
    nb = cfg.hist_nbins; Jmax = cfg.hist_Jmax; scale = nb / Jmax
    coefs = Float64[c for (c, _) in terms]; Gs = [G for (_, G) in terms]; nterm = length(terms)
    Threads.@threads for (ti, rng) in collect(enumerate(chunks))
        acc = accs[ti]
        @inbounds for e in rng
            F = I3
            for t in 1:nterm
                F += coefs[t] * SM3d(Gs[t][e])
            end
            J = det(F)
            δ = Float64(δL[e])
            acc.n += 1; acc.sumJ += J
            acc.minJ = min(acc.minJ, J); acc.maxJ = max(acc.maxJ, J)
            if J > 0
                acc.sumJinv += 1 / J; acc.nJinv += 1
                lnJ = log(J); acc.sum_lnJ += lnJ; acc.n_lnJ += 1
            else
                acc.nJle0 += 1
                lnJ = NaN
            end
            acc.minsv = min(acc.minsv, min_singular_value(F))
            # histogram for percentiles
            if J < 0
                acc.hunder += 1
            elseif J >= Jmax
                acc.hover += 1
            else
                acc.hist[floor(Int, J * scale) + 1] += 1
            end
            # CDF bins: idx = first i with Jstar[i] >= J  (J < Jstar[i'] for all i' >= idx)
            idx = searchsortedfirst(Jstar, J)
            acc.cdf_cnt[idx] += 1; acc.cdf_sumJ[idx] += J
            dpos = δ > 0; jlt1 = J < 1
            acc.ndpos += dpos; acc.nJlt1 += jlt1
            if dpos
                jlt1 ? (acc.n_pp += 1) : (acc.n_mp += 1)
            else
                jlt1 ? (acc.n_pm += 1) : (acc.n_mm += 1)
            end
            # shear focusing near δ_L = 0 : −ln J vs T:T in nested symmetric bins
            aδ = abs(δ)
            x = Float64(TT[e])
            for b in eachindex(widths)
                aδ < widths[b] || continue
                if J > 0
                    y = -lnJ
                    acc.sh_n[b] += 1; acc.sh_x[b] += x; acc.sh_y[b] += y
                    acc.sh_xx[b] += x * x; acc.sh_xy[b] += x * y; acc.sh_yy[b] += y * y
                else
                    acc.sh_excl[b] += 1
                end
            end
        end
    end
    tot = accs[1]
    for t in 2:nt
        merge_accum!(tot, accs[t])
    end
    N = tot.n
    # cumulative CDF: P(J < Jstar[i]) = Σ_{idx<=i} cnt[idx]/N ; F_V = Σ_{idx<=i} sumJ[idx]/N
    cnt_cum = cumsum(tot.cdf_cnt)[1:nJs]; sJ_cum = cumsum(tot.cdf_sumJ)[1:nJs]
    F_M_at_Jstar = cnt_cum ./ N; F_V_at_Jstar = sJ_cum ./ N
    # map to the ρ★ ordering (Jstar ascending == ρ★ descending)
    perm = sortperm(1.0 ./ Jstar)                       # indices of Jstar ordered by increasing ρ★
    F_M = F_M_at_Jstar[perm]; F_V = F_V_at_Jstar[perm]
    # percentiles from histogram
    cum = cumsum(tot.hist)
    pct = similar(cfg.pct)
    for (i, p) in enumerate(cfg.pct)
        target = p * N
        if tot.hunder >= target
            pct[i] = -Inf          # below 0 (reported separately via nJle0)
        else
            k = searchsortedfirst(cum, target - tot.hunder)
            pct[i] = k > nb ? Inf : k / scale   # upper edge of bin k
        end
    end
    # shear regressions
    nb_sh = length(widths)
    slope = fill(NaN, nb_sh); icpt = fill(NaN, nb_sh); rms = fill(NaN, nb_sh)
    for b in 1:nb_sh
        m = tot.sh_n[b]
        m < 3 && continue
        mx = tot.sh_x[b] / m; my = tot.sh_y[b] / m
        vxx = tot.sh_xx[b] / m - mx^2; vxy = tot.sh_xy[b] / m - mx * my; vyy = tot.sh_yy[b] / m - my^2
        s = vxy / vxx
        slope[b] = s; icpt[b] = my - s * mx
        rms[b] = sqrt(max(vyy - s * vxy, 0.0))
    end
    return (n = N, meanJ = tot.sumJ / N, meanJinv = tot.sumJinv / max(tot.nJinv, 1), minJ = tot.minJ, maxJ = tot.maxJ,
            frac_Jle0 = tot.nJle0 / N, min_sv = tot.minsv, mean_lnJ = tot.sum_lnJ / max(tot.n_lnJ, 1),
            P_dpos = tot.ndpos / N, F_M1 = tot.nJlt1 / N,
            F_pp = tot.n_pp / N, F_pm = tot.n_pm / N, F_mp = tot.n_mp / N, F_mm = tot.n_mm / N,
            pct = pct, F_M = F_M, F_V = F_V,
            shear_n = tot.sh_n, shear_excl = tot.sh_excl, shear_slope = slope, shear_icpt = icpt, shear_rms = rms)
end

"""
    taylor_coefficients(A::Vector{Vector{SM3}}) -> (j2, j3, j4, I1, I2, I3, TT)

Exact degree-k parts jₖ of −ln det(I + Σ Aₙ) (Aₙ homogeneous of degree n in the linear
field; pass `A = [A1, A2, A3, A4]`, already multiplied by their growth coefficients) via the
trace–log expansion, plus the invariants of A₁ (I₁ = tr, I₂, I₃ = det) and T:T = tr(A₁²) − I₁²/3.
"""
function taylor_coefficients(A::Vector{Vector{SM3}})
    @assert length(A) >= 2
    n = length(A[1]); has3 = length(A) >= 3; has4 = length(A) >= 4
    j2 = zeros(Float32, n); j3 = zeros(Float32, n); j4 = zeros(Float32, n)
    I1 = zeros(Float32, n); I2 = zeros(Float32, n); I3 = zeros(Float32, n); TT = zeros(Float32, n)
    A1 = A[1]; A2 = A[2]
    Threads.@threads for e in 1:n
        @inbounds begin
            a1 = SM3d(A1[e]); a2 = SM3d(A2[e])
            a1sq = a1 * a1
            t1 = tr(a1); t11 = tr(a1sq)
            I1[e] = t1; I2[e] = 0.5 * (t1^2 - t11); I3[e] = det(a1); TT[e] = t11 - t1^2 / 3
            j2[e] = -(tr(a2) - 0.5 * t11)
            if has3
                a3 = SM3d(A[3][e])
                j3[e] = -(tr(a3) - tr(a1 * a2) + tr(a1sq * a1) / 3)
                if has4
                    a4 = SM3d(A[4][e])
                    j4[e] = -(tr(a4) - 0.5 * tr(a2 * a2) - tr(a1 * a3) + tr(a1sq * a2) - 0.25 * tr(a1sq * a1sq))
                end
            end
        end
    end
    return (j2 = j2, j3 = j3, j4 = j4, I1 = I1, I2 = I2, I3 = I3, TT = TT)
end

"""
    conditional_moments(j, δ, σ, widths; k) -> Vector{NamedTuple}

⟨j/σᵏ⟩ in symmetric bins |δ|/σ < w, with std and count, for the shrinking-bin extrapolation
of ⟨jₖ | δ_L = 0⟩/σᵏ.  Also returns the unconditional mean (w = Inf) and a quadratic
regression intercept of j/σᵏ on ν = δ/σ within |ν| < 0.5 (a + bν + cν² → a at ν = 0).
"""
function conditional_moments(j::Vector{Float32}, δ::Vector{Float32}, σ::Float64, widths::Vector{Float64}; k::Int)
    nw = length(widths)
    nt = Threads.nthreads()
    S1 = zeros(nt, nw + 1); S2 = zeros(nt, nw + 1); C = zeros(Int, nt, nw + 1)
    # quadratic regression accumulators within |ν|<0.5 : normal equations for [1, ν, ν²]
    R = zeros(nt, 3, 3); r = zeros(nt, 3)
    n = length(j); chunks = collect(enumerate(Iterators.partition(1:n, cld(n, nt))))
    norm = σ^k
    Threads.@threads for (ti, rng) in chunks
        @inbounds for e in rng
            y = Float64(j[e]) / norm; ν = Float64(δ[e]) / σ; aν = abs(ν)
            S1[ti, nw + 1] += y; S2[ti, nw + 1] += y * y; C[ti, nw + 1] += 1
            for b in 1:nw
                if aν < widths[b]
                    S1[ti, b] += y; S2[ti, b] += y * y; C[ti, b] += 1
                end
            end
            if aν < 0.5
                v = (1.0, ν, ν * ν)
                for p in 1:3, q in 1:3
                    R[ti, p, q] += v[p] * v[q]
                end
                for p in 1:3
                    r[ti, p] += v[p] * y
                end
            end
        end
    end
    s1 = vec(sum(S1; dims = 1)); s2 = vec(sum(S2; dims = 1)); c = vec(sum(C; dims = 1))
    Rt = dropdims(sum(R; dims = 1); dims = 1); rt = vec(sum(r; dims = 1))
    coef = Rt \ rt
    # standard error of intercept from residual variance
    out = NamedTuple[]
    for b in 1:nw+1
        m = c[b] > 0 ? s1[b] / c[b] : NaN
        sd = c[b] > 1 ? sqrt(max(s2[b] / c[b] - m^2, 0.0)) : NaN
        push!(out, (k = k, width = b <= nw ? widths[b] : Inf, n = c[b], mean = m, std = sd))
    end
    return (bins = out, quad_intercept = coef[1], quad_coefs = coef)
end
