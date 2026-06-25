"""
Differentiable past-lightcone crossing for the faithful exact-growth nLPT.

For each Lagrangian particle the trajectory is
    x(q, a) = q + Σ_k D_k(a) · Ψ_k(q),
with the exact-growth shape fields Ψ = (ψ₁, ψ₂ₑₓ, ψ₃ₐ, ψ₃ᵦ, ψ₃ᵧ) (differentiable in ω)
and growth factors D = (D₁, D₂plus, D₃plusa, D₃plusb, D₃plusc) (constants in ω).  The
crossing scale factor a_cross solves F(a) = |x(q,a) − obs| − χ(a) = 0.

The root-find itself is procedural (bracket + bisection) and is run with the shape
VALUES detached (`@ignore_derivatives`).  The exact implicit-function gradient
∂a_cross/∂Ψ is recovered analytically by taking **one Newton step at the converged
solution**: `a_cross = a* − F(a*,Ψ)/F'`, where a* and F' are stop-gradient constants
and F(a*,Ψ) is differentiable in Ψ.  Since F(a*)=0 the value is unchanged while the
gradient is exactly −∂F/∂Ψ / F' (the IFT result) — so Zygote differentiates the whole
lightcone with no hand-written rrule and no differentiation through the iterations.

`x_obs(q)` and the radial peculiar velocity `v_r(q)` come out differentiable in ω.
The cosmology growth/χ tables are interpolated through cached interpolants built once
per call (the per-call `growth_D*`/`comoving_distance` rebuild their interpolant, which
would dominate the per-particle root-find).
"""

export lightcone_cross_ad, exact_shape_stack

using ChainRulesCore: @ignore_derivatives
using Interpolations: linear_interpolation

# Cached interpolants of the growth factors, χ(a) and f₁(a) over the cosmo a-grid.
struct _CosmoITP{F}
    D::NTuple{5,F}
    chi::F
    f1::F
    amin::Float64
    amax::Float64
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
@inline _Dk(itp::_CosmoITP, a, k) = itp.D[k](_clampa(itp, a))
@inline _chi(itp::_CosmoITP, a)   = itp.chi(_clampa(itp, a))

# Stack the exact-growth shape fields into (N, 3, 5) (row-major flatten of the
# (res,res,res,3) arrays — matches `lagrangian_grid_3d` reshaped the same way).
"""    exact_shape_stack(shapes::Dict) -> (N,3,5) Array (differentiable in ω)"""
function exact_shape_stack(shapes::Dict{String,AbstractArray{T,4}}) where {T}
    keys5 = ("psi_1", "psi_2_ex", "psi_3a_ex", "psi_3b_ex", "psi_3c_ex")
    res = size(shapes["psi_1"], 1); N = res^3
    cols = map(k -> reshape(shapes[k], N, 3), keys5)
    return cat(cols...; dims=3)            # (N,3,5)
end

# ── Forward root-find (procedural, detached) ──────────────────────────────────
function _solve_crossings(Psi::AbstractArray{T,3}, q::AbstractMatrix{T},
                          itp::_CosmoITP, obs::NTuple{3,T},
                          a_far::T, a_near::T; n_scan::Int=16, n_bisect::Int=30) where {T}
    N = size(Psi, 1)
    o1, o2, o3 = obs
    @inline function Fof(i, a)
        x1 = q[i,1]; x2 = q[i,2]; x3 = q[i,3]
        @inbounds for k in 1:5
            d = T(_Dk(itp, a, k))
            x1 += d*Psi[i,1,k]; x2 += d*Psi[i,2,k]; x3 += d*Psi[i,3,k]
        end
        return sqrt((x1-o1)^2 + (x2-o2)^2 + (x3-o3)^2) - T(_chi(itp, a))
    end
    across = fill(T(NaN), N)
    @inbounds for i in 1:N
        a_prev = a_far; F_prev = Fof(i, a_far)
        for s in 1:n_scan
            a_cur = a_far + (a_near - a_far) * s / n_scan
            F_cur = Fof(i, a_cur)
            if (F_prev < 0) != (F_cur < 0)
                lo, hi, Flo = a_prev, a_cur, F_prev
                for _ in 1:n_bisect
                    mid = (lo + hi) / 2; Fm = Fof(i, mid)
                    if (Fm < 0) == (Flo < 0); lo = mid; Flo = Fm; else; hi = mid; end
                end
                across[i] = (lo + hi) / 2
                break
            end
            a_prev = a_cur; F_prev = F_cur
        end
    end
    return across
end

"""
    lightcone_cross_ad(Psi, q, cosmo, observer, a_far, a_near; rsd=false)
        -> (; x_obs, a_cross, v_r, valid)

Differentiable lightcone crossing.  `Psi::(N,3,5)` exact-growth shapes (differentiable
in ω), `q::(N,3)` Lagrangian positions, `observer::(3,)`.  Returns crossing positions
`x_obs::(N,3)` (differentiable in Ψ→ω), `a_cross::(N,)`, the LOS peculiar velocity
`v_r::(N,)` (if `rsd`), and `valid::(N,)` (particles crossing the shell in
`[a_far,a_near]`; non-crossers are parked at `a_near`).
"""
function lightcone_cross_ad(Psi::AbstractArray{T,3}, q::AbstractMatrix{T},
                            cosmo::Cosmology, observer::AbstractVector,
                            a_far::Real, a_near::Real; rsd::Bool=false) where {T}
    N = size(Psi, 1)
    af = T(a_far); an = T(a_near)
    obs = (T(observer[1]), T(observer[2]), T(observer[3]))

    fwd = @ignore_derivatives begin
        itp  = _cosmo_interps(cosmo)
        Psiv = Array(Psi)
        ac0  = _solve_crossings(Psiv, q, itp, obs, af, an)
        valid = .!isnan.(ac0)
        ac   = ifelse.(valid, ac0, an)
        e    = ac .* T(1e-5)
        Dk   = hcat((T.(_Dk.(Ref(itp), ac, k)) for k in 1:5)...)                # (N,5)
        dDk  = hcat(((T.(_Dk.(Ref(itp), ac .+ e, k)) .- T.(_Dk.(Ref(itp), ac .- e, k))) ./ (2 .* e) for k in 1:5)...)
        chi  = T.(_chi.(Ref(itp), ac))
        f1   = T.(itp.f1.(_clampa.(Ref(itp), ac)))
        # Newton derivative F'(a*) at the detached solution
        xstarv = q .+ dropdims(sum(reshape(Dk, N, 1, 5) .* Psiv; dims=3); dims=3)
        xdotv  = dropdims(sum(reshape(dDk, N, 1, 5) .* Psiv; dims=3); dims=3)
        diffv  = xstarv .- reshape(collect(T, obs), 1, 3)
        rv     = sqrt.(sum(abs2, diffv; dims=2))
        drda   = vec(sum(diffv .* xdotv; dims=2)) ./ max.(vec(rv), T(1e-30))
        dchi   = T.(_dchi_da_at.(Ref(cosmo), ac))
        (ac=ac, valid=valid, Dk=Dk, dDk=dDk, chi=chi, f1=f1, Fp=(drda .- dchi))
    end

    obsr = @ignore_derivatives reshape(collect(T, obs), 1, 3)
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
        @ignore_derivatives zeros(T, N)
    end

    return (x_obs=x_obs, a_cross=a_d, v_r=v_r, valid=fwd.valid)
end
