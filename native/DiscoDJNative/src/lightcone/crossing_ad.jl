"""
Differentiable past-lightcone crossing for the faithful exact-growth nLPT.

For each Lagrangian particle the trajectory is
    x(q, a) = q + Σ_k D_k(a) · Ψ_k(q),
with the exact-growth shape fields Ψ (K = 2 for 2LPT → ψ₁, ψ₂ₑₓ; K = 5 for 3LPT →
ψ₁, ψ₂ₑₓ, ψ₃ₐ, ψ₃ᵦ, ψ₃ᵧ; differentiable in ω) and growth factors D = (D₁, D₂plus, …;
the first K).  The crossing scale factor a_cross solves F(a)=|x(q,a)−obs|−χ(a)=0.

The root-find is procedural (a coarse scalar-a sign-change scan then per-particle
bisection) and is run with the shape VALUES detached (`@ignore_derivatives`), fully
**vectorised / backend-agnostic** — it runs natively on `CuArray`s (the growth/χ table
lookups are device row-gathers; geometric a-grid ⇒ O(1) bracket index, no scalar
indexing).  The exact implicit-function gradient ∂a_cross/∂Ψ is recovered by one Newton
step at the converged solution `a_cross = a* − F(a*,Ψ)/F'` (a*, F' stop-grad; F(a*)=0 ⇒
value unchanged, gradient = −∂F/∂Ψ/F').  So Zygote differentiates the whole lightcone
with no hand rrule.  The number of orders K is read from `size(Ψ,3)` throughout.
"""

export lightcone_cross_ad, exact_shape_stack

using ChainRulesCore: @ignore_derivatives
using Interpolations: linear_interpolation

# Cached CPU interpolants of the growth factors / χ(a) for the scalar-a scan.
struct _CosmoITP{F}
    D::NTuple{5,F}; chi::F; f1::F; amin::Float64; amax::Float64
end
function _cosmo_interps(c::Cosmology)
    a = c._a_table
    D = (linear_interpolation(a, c._D1_table),  linear_interpolation(a, c._D2_table),
         linear_interpolation(a, c._D3a_table), linear_interpolation(a, c._D3b_table),
         linear_interpolation(a, c._D3c_table))
    _CosmoITP(D, linear_interpolation(a, c._chi_table), linear_interpolation(a, c._f1_table),
              Float64(a[1]), Float64(a[end]))
end
@inline _clampa(itp::_CosmoITP, a) = clamp(a, oftype(a, itp.amin), oftype(a, itp.amax))

# Copy a host array to the backend of `ref` (constant data → off the AD tape).
_devcopy(ref::AbstractArray, x::AbstractArray{S}) where {S} =
    (y = similar(ref, S, size(x)); copyto!(y, x); y)

# Interpolate the (n,K) growth matrix's rows + extra (n,) columns at a (N-vector); the
# a-grid is geometric (uniform in log a), so the bracket index is O(1) and the lookup a
# device gather.  Returns (Dm::(N,K), tuple of interpolated extra columns).
function _interp_growth(atab, Dmat, vcols::Tuple, a::AbstractVector, logamin, dloga, n::Int)
    jf = (log10.(a) .- logamin) ./ dloga
    j  = clamp.(floor.(Int, jf), 0, n - 2)
    j1 = j .+ 1; j2 = j .+ 2
    aj = atab[j1]; aj1 = atab[j2]
    fr = (a .- aj) ./ (aj1 .- aj); frm = reshape(fr, :, 1)
    Dm = Dmat[j1, :] .* (1 .- frm) .+ Dmat[j2, :] .* frm
    vc = map(c -> c[j1] .* (1 .- fr) .+ c[j2] .* fr, vcols)
    return Dm, vc
end

# Stack the exact-growth shape fields into (N, 3, K) — K = 1 (1LPT/Zel'dovich → ψ₁),
# 2 (2LPT → ψ₁,ψ₂ₑₓ) or 5 (3LPT → ψ₁,ψ₂ₑₓ,ψ₃ₐ,ψ₃ᵦ,ψ₃ᵧ).
"""    exact_shape_stack(shapes::Dict) -> (N,3,K) Array (differentiable in ω)"""
function exact_shape_stack(shapes::Dict{String,<:AbstractArray{T,4}}) where {T}
    ks = haskey(shapes, "psi_3a_ex") ? ("psi_1", "psi_2_ex", "psi_3a_ex", "psi_3b_ex", "psi_3c_ex") :
         haskey(shapes, "psi_2_ex")  ? ("psi_1", "psi_2_ex") : ("psi_1",)
    res = size(shapes["psi_1"], 1); N = res^3
    cols = map(k -> reshape(shapes[k], N, 3), ks)
    return cat(cols...; dims=3)
end

# ── Vectorised, device-aware forward root-find (detached) ─────────────────────
function _crossing_forward(Psi::AbstractArray{T,3}, q::AbstractMatrix{T},
                           cosmo::Cosmology, obs::NTuple{3,T}, af::T, an::T;
                           n_scan::Int=16, n_bisect::Int=30) where {T}
    N = size(Psi, 1); K = size(Psi, 3)
    itp = _cosmo_interps(cosmo)
    atabh = cosmo._a_table; n = length(atabh)
    logamin = log10(T(atabh[1])); dloga = (log10(T(atabh[end])) - logamin) / (n - 1)
    aD = _devcopy(Psi, atabh)
    Ktab = (cosmo._D1_table, cosmo._D2_table, cosmo._D3a_table, cosmo._D3b_table, cosmo._D3c_table)
    Dmat = _devcopy(Psi, reduce(hcat, Ktab[1:K]))            # (n,K) on device
    chic = _devcopy(Psi, cosmo._chi_table); f1c = _devcopy(Psi, cosmo._f1_table)
    o1, o2, o3 = obs
    q1 = q[:, 1]; q2 = q[:, 2]; q3 = q[:, 3]
    P1 = Psi[:, 1, :]; P2 = Psi[:, 2, :]; P3 = Psi[:, 3, :]   # (N,K) per component

    function Fscalar(a)
        Dv = _devcopy(Psi, T[itp.D[k](_clampa(itp, a)) for k in 1:K]); ca = T(itp.chi(_clampa(itp, a)))
        dx = q1 .+ (P1 * Dv) .- o1; dy = q2 .+ (P2 * Dv) .- o2; dz = q3 .+ (P3 * Dv) .- o3
        return sqrt.(dx.^2 .+ dy.^2 .+ dz.^2) .- ca
    end
    function Fvec(a)
        Dm, (ca,) = _interp_growth(aD, Dmat, (chic,), a, logamin, dloga, n)
        dx = q1 .+ vec(sum(P1 .* Dm; dims=2)) .- o1
        dy = q2 .+ vec(sum(P2 .* Dm; dims=2)) .- o2
        dz = q3 .+ vec(sum(P3 .* Dm; dims=2)) .- o3
        return sqrt.(dx.^2 .+ dy.^2 .+ dz.^2) .- ca
    end

    lo = fill!(similar(Psi, T, N), af); hi = fill!(similar(Psi, T, N), an)
    found = fill!(similar(Psi, Bool, N), false)
    a_prev = af; F_prev = Fscalar(af)
    for s in 1:n_scan
        a_cur = af + (an - af) * s / n_scan
        F_cur = Fscalar(a_cur)
        nb = ((F_prev .< 0) .!= (F_cur .< 0)) .& .!found
        lo = ifelse.(nb, a_prev, lo); hi = ifelse.(nb, a_cur, hi); found = found .| nb
        a_prev = a_cur; F_prev = F_cur
    end
    Flo = Fvec(lo)
    for _ in 1:n_bisect
        mid  = (lo .+ hi) ./ 2
        Fm   = Fvec(mid)
        same = (Fm .< 0) .== (Flo .< 0)
        lo = ifelse.(same, mid, lo); Flo = ifelse.(same, Fm, Flo); hi = ifelse.(same, hi, mid)
    end
    ac = ifelse.(found, (lo .+ hi) ./ 2, an)

    Dk, (chi, f1) = _interp_growth(aD, Dmat, (chic, f1c), ac, logamin, dloga, n)
    e   = ac .* T(1e-5)
    Dp, _  = _interp_growth(aD, Dmat, (), ac .+ e, logamin, dloga, n)
    Dm2, _ = _interp_growth(aD, Dmat, (), ac .- e, logamin, dloga, n)
    dDk = (Dp .- Dm2) ./ (2 .* reshape(e, :, 1))
    xs1 = q1 .+ vec(sum(P1 .* Dk; dims=2)); xs2 = q2 .+ vec(sum(P2 .* Dk; dims=2)); xs3 = q3 .+ vec(sum(P3 .* Dk; dims=2))
    xd1 = vec(sum(P1 .* dDk; dims=2)); xd2 = vec(sum(P2 .* dDk; dims=2)); xd3 = vec(sum(P3 .* dDk; dims=2))
    d1 = xs1 .- o1; d2 = xs2 .- o2; d3 = xs3 .- o3
    r  = sqrt.(d1.^2 .+ d2.^2 .+ d3.^2)
    drda = (d1 .* xd1 .+ d2 .* xd2 .+ d3 .* xd3) ./ max.(r, T(1e-30))
    Om = Omega_m(cosmo); Ok = cosmo.Omega_k; w0 = cosmo.w0; wa = cosmo.wa; Ode0 = 1 - Om - Ok
    Ea(a) = sqrt(Om * a^(-3) + Ok * a^(-2) + Ode0 * a^(-3 * (1 + w0 + wa)) * exp(-3 * wa * (1 - a)))
    Fp = drda .- (.-T(2997.92458) ./ (ac.^2 .* Ea.(ac)))
    return (ac=ac, valid=found, Dk=Dk, dDk=dDk, chi=chi, f1=f1, Fp=Fp)
end

"""
    lightcone_cross_ad(Psi, q, cosmo, observer, a_far, a_near; rsd=false)
        -> (; x_obs, a_cross, v_r, valid)

Differentiable lightcone crossing (see module docstring).  `Psi::(N,3,K)`, `q::(N,3)`
on any backend; the result is differentiable in Ψ (→ ω).
"""
function lightcone_cross_ad(Psi::AbstractArray{T,3}, q::AbstractMatrix{T},
                            cosmo::Cosmology, observer::AbstractVector,
                            a_far::Real, a_near::Real; rsd::Bool=false) where {T}
    N = size(Psi, 1); K = size(Psi, 3)
    af = T(a_far); an = T(a_near)
    obs = (T(observer[1]), T(observer[2]), T(observer[3]))
    fwd = @ignore_derivatives _crossing_forward(Psi, q, cosmo, obs, af, an)

    obsr  = @ignore_derivatives permutedims(_devcopy(Psi, T[obs...]))   # (1,3) on backend
    xstar = q .+ dropdims(sum(reshape(fwd.Dk, N, 1, K) .* Psi; dims=3); dims=3)
    diff  = xstar .- obsr
    r     = sqrt.(sum(abs2, diff; dims=2))
    F     = vec(r) .- fwd.chi
    a_d   = fwd.ac .- F ./ fwd.Fp
    xdot  = dropdims(sum(reshape(fwd.dDk, N, 1, K) .* Psi; dims=3); dims=3)
    x_obs = xstar .+ xdot .* (a_d .- fwd.ac)

    v_r = if rsd
        rhat = diff ./ max.(r, T(1e-30))
        fk   = @ignore_derivatives fwd.f1 .* fwd.Dk
        vvec = dropdims(sum(reshape(fk, N, 1, K) .* Psi; dims=3); dims=3)
        vec(sum(vvec .* rhat; dims=2))
    else
        @ignore_derivatives fill!(similar(r, N), zero(T))
    end
    return (x_obs=x_obs, a_cross=a_d, v_r=v_r, valid=fwd.valid)
end
