"""
Differentiable past-lightcone crossing for the faithful exact-growth nLPT.

For each Lagrangian particle the trajectory is
    x(q, a) = q + Σ_k D_k(a) · Ψ_k(q),
with the exact-growth shape fields Ψ = (ψ₁, ψ₂ₑₓ, ψ₃ₐ, ψ₃ᵦ, ψ₃ᵧ) (differentiable in ω)
and growth factors D = (D₁, D₂plus, D₃plusa, D₃plusb, D₃plusc).  The crossing scale factor
a_cross solves F(a) = |x(q,a) − obs| − χ(a) = 0.

The root-find is procedural (a coarse scalar-a sign-change scan then per-particle
bisection) and is run with the shape VALUES detached (`@ignore_derivatives`), fully
**vectorised / backend-agnostic** — it runs natively on `CuArray`s (the table lookups
are device gathers, no scalar indexing).  The exact implicit-function gradient
∂a_cross/∂Ψ is recovered by one Newton step at the converged solution
`a_cross = a* − F(a*,Ψ)/F'` (a*, F' stop-grad; F(a*)=0 ⇒ value unchanged, gradient =
−∂F/∂Ψ/F').  So Zygote differentiates the whole lightcone with no hand rrule.

`x_obs(q)` and the radial peculiar velocity `v_r(q)` come out differentiable in ω.
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

# Copy a host table to the backend of `ref` (constant data → off the AD tape).
_devcopy(ref::AbstractArray, x::AbstractVector{S}) where {S} =
    (y = similar(ref, S, length(x)); copyto!(y, x); y)

# Per-particle linear interpolation of table columns at a (N-vector); the a-grid is
# geometric (uniform in log a), so the bracket index is O(1) and the lookup a gather.
function _interp_cols(atab, a::AbstractVector, logamin, dloga, n::Int, cols::Tuple)
    jf = (log10.(a) .- logamin) ./ dloga
    j  = clamp.(floor.(Int, jf), 0, n - 2)
    j1 = j .+ 1; j2 = j .+ 2
    aj = atab[j1]; aj1 = atab[j2]
    fr = (a .- aj) ./ (aj1 .- aj)
    return map(c -> c[j1] .* (1 .- fr) .+ c[j2] .* fr, cols)
end

# Stack the exact-growth shape fields into (N, 3, 5).
"""    exact_shape_stack(shapes::Dict) -> (N,3,5) Array (differentiable in ω)"""
function exact_shape_stack(shapes::Dict{String,<:AbstractArray{T,4}}) where {T}
    keys5 = ("psi_1", "psi_2_ex", "psi_3a_ex", "psi_3b_ex", "psi_3c_ex")
    res = size(shapes["psi_1"], 1); N = res^3
    cols = map(k -> reshape(shapes[k], N, 3), keys5)
    return cat(cols...; dims=3)            # (N,3,5)
end

# ── Vectorised, device-aware forward root-find (detached) ─────────────────────
function _crossing_forward(Psi::AbstractArray{T,3}, q::AbstractMatrix{T},
                           cosmo::Cosmology, obs::NTuple{3,T}, af::T, an::T;
                           n_scan::Int=16, n_bisect::Int=30) where {T}
    N = size(Psi, 1)
    itp = _cosmo_interps(cosmo)
    atabh = cosmo._a_table; n = length(atabh)
    logamin = log10(T(atabh[1])); dloga = (log10(T(atabh[end])) - logamin) / (n - 1)
    aD   = _devcopy(Psi, atabh)
    Dcol = map(t -> _devcopy(Psi, t),
               (cosmo._D1_table, cosmo._D2_table, cosmo._D3a_table, cosmo._D3b_table, cosmo._D3c_table))
    chic = _devcopy(Psi, cosmo._chi_table); f1c = _devcopy(Psi, cosmo._f1_table)
    o1, o2, o3 = obs
    q1 = q[:, 1]; q2 = q[:, 2]; q3 = q[:, 3]
    P1 = Psi[:, 1, :]; P2 = Psi[:, 2, :]; P3 = Psi[:, 3, :]   # (N,5) per component

    # F at a SCALAR a (growth from the cached host interpolants → device 5-vector).
    function Fscalar(a)
        Dv = _devcopy(Psi, T[itp.D[k](_clampa(itp, a)) for k in 1:5])
        ca = T(itp.chi(_clampa(itp, a)))
        dx = q1 .+ (P1 * Dv) .- o1; dy = q2 .+ (P2 * Dv) .- o2; dz = q3 .+ (P3 * Dv) .- o3
        return sqrt.(dx.^2 .+ dy.^2 .+ dz.^2) .- ca
    end
    # F at PER-PARTICLE a (gather interpolation → (N,5) growth).
    function Fvec(a)
        ic = _interp_cols(aD, a, logamin, dloga, n, (Dcol..., chic))
        Dm = hcat(ic[1], ic[2], ic[3], ic[4], ic[5]); ca = ic[6]
        dx = q1 .+ vec(sum(P1 .* Dm; dims=2)) .- o1
        dy = q2 .+ vec(sum(P2 .* Dm; dims=2)) .- o2
        dz = q3 .+ vec(sum(P3 .* Dm; dims=2)) .- o3
        return sqrt.(dx.^2 .+ dy.^2 .+ dz.^2) .- ca
    end

    # coarse scan → per-particle bracket [lo, hi] of the first sign change
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
    # bisection (vectorised; keeps the sign of F at lo)
    Flo = Fvec(lo)
    for _ in 1:n_bisect
        mid  = (lo .+ hi) ./ 2
        Fm   = Fvec(mid)
        same = (Fm .< 0) .== (Flo .< 0)
        lo = ifelse.(same, mid, lo); Flo = ifelse.(same, Fm, Flo); hi = ifelse.(same, hi, mid)
    end
    ac = ifelse.(found, (lo .+ hi) ./ 2, an)        # non-crossers parked at a_near

    # growth, dD/da, χ, f₁ at a*, and the Newton derivative F'(a*)
    Dk7 = _interp_cols(aD, ac, logamin, dloga, n, (Dcol..., chic, f1c))
    Dk  = hcat(Dk7[1], Dk7[2], Dk7[3], Dk7[4], Dk7[5]); chi = Dk7[6]; f1 = Dk7[7]
    e   = ac .* T(1e-5)
    Dp  = _interp_cols(aD, ac .+ e, logamin, dloga, n, Dcol)
    Dm2 = _interp_cols(aD, ac .- e, logamin, dloga, n, Dcol)
    dDk = hcat(ntuple(k -> (Dp[k] .- Dm2[k]) ./ (2 .* e), 5)...)
    xs1 = q1 .+ vec(sum(P1 .* Dk; dims=2)); xs2 = q2 .+ vec(sum(P2 .* Dk; dims=2)); xs3 = q3 .+ vec(sum(P3 .* Dk; dims=2))
    xd1 = vec(sum(P1 .* dDk; dims=2)); xd2 = vec(sum(P2 .* dDk; dims=2)); xd3 = vec(sum(P3 .* dDk; dims=2))
    d1 = xs1 .- o1; d2 = xs2 .- o2; d3 = xs3 .- o3
    r  = sqrt.(d1.^2 .+ d2.^2 .+ d3.^2)
    drda = (d1 .* xd1 .+ d2 .* xd2 .+ d3 .* xd3) ./ max.(r, T(1e-30))
    # dχ/da = −(c/H₀)/(a²E(a)); inline E with captured SCALAR cosmo params (GPU-safe).
    Om = Omega_m(cosmo); Ok = cosmo.Omega_k; w0 = cosmo.w0; wa = cosmo.wa; Ode0 = 1 - Om - Ok
    Ea(a) = sqrt(Om * a^(-3) + Ok * a^(-2) + Ode0 * a^(-3 * (1 + w0 + wa)) * exp(-3 * wa * (1 - a)))
    Fp = drda .- (.-T(2997.92458) ./ (ac.^2 .* Ea.(ac)))
    return (ac=ac, valid=found, Dk=Dk, dDk=dDk, chi=chi, f1=f1, Fp=Fp)
end

"""
    lightcone_cross_ad(Psi, q, cosmo, observer, a_far, a_near; rsd=false)
        -> (; x_obs, a_cross, v_r, valid)

Differentiable lightcone crossing (see module docstring).  `Psi::(N,3,5)`, `q::(N,3)`
on any backend; the result is differentiable in Ψ (→ ω).
"""
function lightcone_cross_ad(Psi::AbstractArray{T,3}, q::AbstractMatrix{T},
                            cosmo::Cosmology, observer::AbstractVector,
                            a_far::Real, a_near::Real; rsd::Bool=false) where {T}
    N = size(Psi, 1)
    af = T(a_far); an = T(a_near)
    obs = (T(observer[1]), T(observer[2]), T(observer[3]))
    fwd = @ignore_derivatives _crossing_forward(Psi, q, cosmo, obs, af, an)

    obsr  = @ignore_derivatives permutedims(_devcopy(Psi, T[obs...]))   # (1,3) on backend
    xstar = q .+ dropdims(sum(reshape(fwd.Dk, N, 1, 5) .* Psi; dims=3); dims=3)   # value = crossing pos
    diff  = xstar .- obsr
    r     = sqrt.(sum(abs2, diff; dims=2))
    F     = vec(r) .- fwd.chi
    a_d   = fwd.ac .- F ./ fwd.Fp                                                 # IFT-corrected a_cross
    xdot  = dropdims(sum(reshape(fwd.dDk, N, 1, 5) .* Psi; dims=3); dims=3)       # dx/da at a*
    x_obs = xstar .+ xdot .* (a_d .- fwd.ac)                                      # value = xstar; IFT gradient

    v_r = if rsd
        rhat = diff ./ max.(r, T(1e-30))
        fk   = @ignore_derivatives fwd.f1 .* fwd.Dk                               # f₁·D_k  (N,5)
        vvec = dropdims(sum(reshape(fk, N, 1, 5) .* Psi; dims=3); dims=3)
        vec(sum(vvec .* rhat; dims=2))
    else
        @ignore_derivatives fill!(similar(r, N), zero(T))
    end
    return (x_obs=x_obs, a_cross=a_d, v_r=v_r, valid=fwd.valid)
end
