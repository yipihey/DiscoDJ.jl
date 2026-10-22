# Regression machinery for the one-scale vs two-scale (cube vs cube+parent) locality test.
# Included after MS.jl by multiscale/bin/regress.jl.
module Reg

using LinearAlgebra, Statistics, StaticArrays, Printf
import ..MS
const M = MS.M

struct Group
    AC::Matrix{Float32}      # (6,n) cube-averaged initial deformation (eps = 1 units)
    AP::Matrix{Float32}      # (6,n) parent
    J::Vector{Float32}       # cube Jacobian of the target method
    name::String
end
nrows(g::Group) = length(g.J)

const CUTS = (0.0, 0.1, 0.2)
const KINDS = (:ln, :J)
cutname(c) = c == 0.0 ? "none" : "J>$(c)"
# row mask for (kind,cut): ln-target needs J>0 always
inmask(kind, cut, J) = kind == :ln ? (cut == 0.0 ? J > 0 : J > cut) : (cut == 0.0 ? true : J > cut)
yval(kind, J) = kind == :ln ? log(Float64(J)) : Float64(J)

# mask identity shared between (kind,cut) combos:  cut=0 -> :pos (ln) / :all (J) ; 0.1 -> :c01 ; 0.2 -> :c02
maskid(kind, cut) = cut == 0.0 ? (kind == :ln ? :pos : :all) : (cut == 0.1 ? :c01 : :c02)
const MASKIDS = (:all, :pos, :c01, :c02)
maskpass(m, J) = m == :all ? true : m == :pos ? J > 0 : m == :c01 ? J > 0.1 : J > 0.2

mutable struct NE      # normal equations of one design on one mask
    p::Int; n::Int
    XtX::Matrix{Float64}; Xty_ln::Vector{Float64}; Xty_J::Vector{Float64}
    sy_ln::Float64; syy_ln::Float64; sy_J::Float64; syy_J::Float64
end
NE(p) = NE(p, 0, zeros(p, p), zeros(p), zeros(p), 0.0, 0.0, 0.0, 0.0)

function addrows!(ne::NE, X::Matrix{Float64}, J::AbstractVector{Float32}, m::Symbol)
    rows = [r for r in eachindex(J) if maskpass(m, J[r])]
    isempty(rows) && return
    Xm = X[rows, :]
    ne.n += length(rows)
    BLAS.syrk!('U', 'T', 1.0, Xm, 1.0, ne.XtX)
    Jm = Float64.(J[rows])
    ne.Xty_J .+= Xm' * Jm; ne.sy_J += sum(Jm); ne.syy_J += sum(abs2, Jm)
    if m != :all
        yl = log.(Jm)
        ne.Xty_ln .+= Xm' * yl; ne.sy_ln += sum(yl); ne.syy_ln += sum(abs2, yl)
    end
end
function finalize!(ne::NE)
    for j in 1:ne.p, i in j+1:ne.p
        ne.XtX[i, j] = ne.XtX[j, i]
    end
    return ne
end

"Solve the (column-scaled, tiny-ridge) normal equations restricted to columns `idx`."
function solve_ne(ne::NE, kind::Symbol, idx; ridge = 1e-10)
    A = ne.XtX[idx, idx]; b = (kind == :ln ? ne.Xty_ln : ne.Xty_J)[idx]
    syy = kind == :ln ? ne.syy_ln : ne.syy_J; sy = kind == :ln ? ne.sy_ln : ne.sy_J; n = ne.n
    s = sqrt.(max.(diag(A) ./ n, 1e-300))
    As = Symmetric(A ./ (s * s') + ridge * I); bs = b ./ s
    βs = try
        cholesky(As) \ bs
    catch
        ev = eigen(As); keep = ev.values .> 1e-13 * maximum(ev.values)
        ev.vectors[:, keep] * ((ev.vectors[:, keep]' * bs) ./ ev.values[keep])
    end
    β = βs ./ s
    sse = syy - 2 * dot(β, b) + dot(β, A * β)
    sres = sy - dot(β, A[1, :])
    vtrain = max(sse / n - (sres / n)^2, 0.0)
    vy = syy / n - (sy / n)^2
    return (β = β, vtrain = vtrain, vy = vy, n = n)
end

mutable struct TS
    n::Int; sr1::Float64; srr1::Float64; sr2::Float64; srr2::Float64; sy::Float64; syy::Float64
end
TS() = TS(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
mutable struct CS
    nall::Int; nM::Int; npm::Int; nmp::Int; ndpos::Int
    nM1::Int; npm1::Int; nmp1::Int; nM2::Int; npm2::Int; nmp2::Int
end
CS() = CS(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

const CHUNK = 100_000

"""
    run_case(train, test, ε; flex) -> (fits, stats, mons)

fits[(model,kind,cut,scale)] = (β, vtrain, vy, n); stats[(model,kind,cut)] = Vector{(TS,CS)} over the test groups.
model ∈ :inv (26-term invariant basis, scale 1 = first 11 columns) | :flex (raw-component monomials, deg ≤ 3).
"""
function run_case(train::Vector{Group}, test::Vector{Group}, ε::Float64; flex::Bool = false, log = stdout)
    mons1 = MS.monomials(6, 3); mons2 = MS.monomials(12, 3)
    ne_inv = Dict(m => NE(26) for m in MASKIDS)
    ne_f1 = flex ? NE(length(mons1)) : nothing; ne_f2 = flex ? NE(length(mons2)) : nothing
    for g in train
        n = nrows(g)
        for lo in 1:CHUNK:n
            rng = lo:min(lo + CHUNK - 1, n)
            Jc = g.J[rng]
            X = Matrix{Float64}(undef, length(rng), 26); MS.design!(X, g.AC, g.AP, rng, ε, :inv)
            for m in MASKIDS
                addrows!(ne_inv[m], X, Jc, m)
            end
            if flex
                k = [r for r in eachindex(Jc) if Jc[r] > 0.1]
                for (ne, mons, kd) in ((ne_f1, mons1, :flex1), (ne_f2, mons2, :flex2))
                    Xf = Matrix{Float64}(undef, length(rng), length(mons)); MS.design!(Xf, g.AC, g.AP, rng, ε, kd; mons)
                    addrows!(ne, Xf, Jc, :c01)
                end
            end
        end
        println(log, "    train group $(g.name): done"); flush(log)
    end
    for m in MASKIDS; finalize!(ne_inv[m]); end
    if flex; finalize!(ne_f1); finalize!(ne_f2); end
    fits = Dict{Tuple,Any}()
    for kind in KINDS, cut in CUTS
        ne = ne_inv[maskid(kind, cut)]
        fits[(:inv, kind, cut, 1)] = solve_ne(ne, kind, 1:11)
        fits[(:inv, kind, cut, 2)] = solve_ne(ne, kind, 1:26)
        if flex && cut == 0.1
            fits[(:flex, kind, cut, 1)] = solve_ne(ne_f1, kind, 1:length(mons1); ridge = 1e-9)
            fits[(:flex, kind, cut, 2)] = solve_ne(ne_f2, kind, 1:length(mons2); ridge = 1e-9)
        end
    end
    # ---- test pass
    stats = Dict{Tuple,Vector{Tuple{TS,CS}}}()
    for key in keys(fits)
        k3 = key[1:3]
        haskey(stats, k3) || (stats[k3] = [(TS(), CS()) for _ in test])
    end
    for (gi, g) in enumerate(test)
        n = nrows(g)
        for lo in 1:CHUNK:n
            rng = lo:min(lo + CHUNK - 1, n)
            Jc = g.J[rng]
            δ = [-ε * (Float64(g.AC[1, e]) + Float64(g.AC[2, e]) + Float64(g.AC[3, e])) for e in rng]
            X = Matrix{Float64}(undef, length(rng), 26); MS.design!(X, g.AC, g.AP, rng, ε, :inv)
            Xf = Dict{Int,Matrix{Float64}}()
            if flex
                Xf[1] = Matrix{Float64}(undef, length(rng), length(mons1)); MS.design!(Xf[1], g.AC, g.AP, rng, ε, :flex1; mons = mons1)
                Xf[2] = Matrix{Float64}(undef, length(rng), length(mons2)); MS.design!(Xf[2], g.AC, g.AP, rng, ε, :flex2; mons = mons2)
            end
            for k3 in keys(stats)
                model, kind, cut = k3
                f1 = fits[(model, kind, cut, 1)]; f2 = fits[(model, kind, cut, 2)]
                p1 = model == :inv ? X[:, 1:11] * f1.β : Xf[1] * f1.β
                p2 = model == :inv ? X * f2.β : Xf[2] * f2.β
                ts, cs = stats[k3][gi]
                thr = kind == :ln ? 0.0 : 1.0
                for r in eachindex(Jc)
                    J = Jc[r]
                    if inmask(kind, cut, J)
                        y = yval(kind, J)
                        r1 = y - p1[r]; r2 = y - p2[r]
                        ts.n += 1; ts.sr1 += r1; ts.srr1 += r1^2; ts.sr2 += r2; ts.srr2 += r2^2; ts.sy += y; ts.syy += y^2
                    end
                    cs.nall += 1
                    jlt1 = J < 1; dneg = δ[r] < 0; dpos = !dneg
                    cs.nM += jlt1; cs.npm += (dneg & jlt1); cs.nmp += (dpos & (J > 1)); cs.ndpos += (δ[r] > 0)
                    a1 = p1[r] < thr; a2 = p2[r] < thr
                    cs.nM1 += a1; cs.npm1 += (dneg & a1); cs.nmp1 += (dpos & (p1[r] > thr))
                    cs.nM2 += a2; cs.npm2 += (dneg & a2); cs.nmp2 += (dpos & (p2[r] > thr))
                end
            end
        end
        println(log, "    test group $(g.name): done"); flush(log)
    end
    return fits, stats, (mons1, mons2)
end

"variance helpers from a TS"
var1(t::TS) = t.srr1 / t.n - (t.sr1 / t.n)^2
var2(t::TS) = t.srr2 / t.n - (t.sr2 / t.n)^2
vary(t::TS) = t.syy / t.n - (t.sy / t.n)^2

function pooled(v::Vector{Tuple{TS,CS}})
    t = TS(); c = CS()
    for (ti, ci) in v
        for f in fieldnames(TS); setfield!(t, f, getfield(t, f) + getfield(ti, f)); end
        for f in fieldnames(CS); setfield!(c, f, getfield(c, f) + getfield(ci, f)); end
    end
    return t, c
end

end # module
