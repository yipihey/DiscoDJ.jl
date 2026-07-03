"""
Differentiable past-lightcone crossing for the faithful exact-growth nLPT.

For each Lagrangian particle the trajectory is
    x(q, a) = q + Σ_k D_k(a) · Ψ_k(q),
with the exact-growth shape fields Ψ (K = 2 for 2LPT → ψ₁, ψ₂ₑₓ; K = 5 for 3LPT →
ψ₁, ψ₂ₑₓ, ψ₃ₐ, ψ₃ᵦ, ψ₃ᵧ; differentiable in ω) and growth factors D = (D₁, D₂plus, …;
the first K).  The crossing scale factor a_cross solves F(a)=|x(q,a)−obs|−χ(a)=0.

The root-find is procedural (per-particle sign-change scan then bisection) and is run
with the shape VALUES detached (`@ignore_derivatives`), **fused into a single
KernelAbstractions kernel** (one launch on CPU or CUDA): each particle scans + bisects
entirely in registers against the shared geometric-a growth/χ tables (O(1) log-a index),
with no intermediate global-memory traffic — this is the dominant cost of the cheap
(1LPT) path, so collapsing the prior ~47-iteration × ~8-kernel storm matters most there.
The exact implicit-function gradient ∂a_cross/∂Ψ is recovered by one Newton
step at the converged solution `a_cross = a* − F(a*,Ψ)/F'` (a*, F' stop-grad; F(a*)=0 ⇒
value unchanged, gradient = −∂F/∂Ψ/F').  So Zygote differentiates the whole lightcone
with no hand rrule.  The number of orders K is read from `size(Ψ,3)` throughout.
"""

export lightcone_cross_ad, exact_shape_stack

using ChainRulesCore: @ignore_derivatives
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize

# Copy a host array to the backend of `ref` (constant data → off the AD tape).
_devcopy(ref::AbstractArray, x::AbstractArray{S}) where {S} =
    (y = similar(ref, S, size(x)); copyto!(y, x); y)

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

# ── Fused per-particle forward root-find (one KA kernel, CPU + CUDA) ───────────
# Each particle independently scans + bisects F(a)=|q+ΣP_kD_k(a)−obs|−χ(a)=0 entirely
# in registers, looking up the shared geometric-a growth/χ tables (Dmat (n,K), chitab,
# f1tab) by O(1) log-a index.  Replaces the previous ~47-iteration × ~8-kernel launch
# storm (+17 host→device syncs): one launch, no intermediate global-memory traffic —
# this is the dominant cost of the cheap (1LPT) path.  Detached (inside
# `@ignore_derivatives`); the IFT gradient is recovered analytically downstream.

# Linear interp of one geometric-a table column at scalar `a` (Nyquist-flat ends).
@inline function _tabidx(a, logamin, dloga, n, atab)
    jf = (log10(a) - logamin) / dloga
    j  = clamp(unsafe_trunc(Int, floor(jf)), 0, n - 2)
    j1 = j + 1; j2 = j + 2
    fr = (a - atab[j1]) / (atab[j2] - atab[j1])
    return j1, j2, fr
end

# F(a) = |q + Σ_k P_k D_k(a) − obs| − χ(a) for particle p (no closure → GPU-clean).
@inline function _Fcross(a, p, q1, q2, q3, o1, o2, o3, Psi, Dmat, chitab, K, n, logamin, dloga, atab)
    j1, j2, fr = _tabidx(a, logamin, dloga, n, atab)
    s1 = q1 - o1; s2 = q2 - o2; s3 = q3 - o3
    @inbounds for k in 1:K
        Dkk = Dmat[j1, k] * (1 - fr) + Dmat[j2, k] * fr
        s1 += Psi[p, 1, k] * Dkk; s2 += Psi[p, 2, k] * Dkk; s3 += Psi[p, 3, k] * Dkk
    end
    @inbounds ca = chitab[j1] * (1 - fr) + chitab[j2] * fr
    return sqrt(s1 * s1 + s2 * s2 + s3 * s3) - ca
end

@kernel function _cross_kernel!(ac, valid, Dk, dDk, chi, f1, Fp,
        @Const(q), @Const(Psi), @Const(atab), @Const(Dmat), @Const(chitab), @Const(f1tab),
        K::Int, n::Int, logamin, dloga, af, an, o1, o2, o3,
        Om, Ok, Ode0, w0, wa, nscan::Int, nbisect::Int)
    p = @index(Global)
    @inbounds if p <= size(q, 1)
        T = eltype(q)
        q1 = q[p, 1]; q2 = q[p, 2]; q3 = q[p, 3]
        # Scan [af,an] for the first sign change → bracket [lo,hi].
        lo = af; hi = an; found = false
        a_prev = af; F_prev = _Fcross(af, p, q1,q2,q3, o1,o2,o3, Psi, Dmat, chitab, K, n, logamin, dloga, atab)
        for s in 1:nscan
            a_cur = af + (an - af) * s / nscan
            F_cur = _Fcross(a_cur, p, q1,q2,q3, o1,o2,o3, Psi, Dmat, chitab, K, n, logamin, dloga, atab)
            if !found && ((F_prev < 0) != (F_cur < 0))
                lo = a_prev; hi = a_cur; found = true
            end
            a_prev = a_cur; F_prev = F_cur
        end
        # Bisect the bracket.
        Flo = _Fcross(lo, p, q1,q2,q3, o1,o2,o3, Psi, Dmat, chitab, K, n, logamin, dloga, atab)
        for _ in 1:nbisect
            mid = (lo + hi) / 2
            Fm  = _Fcross(mid, p, q1,q2,q3, o1,o2,o3, Psi, Dmat, chitab, K, n, logamin, dloga, atab)
            if (Fm < 0) == (Flo < 0); lo = mid; Flo = Fm; else; hi = mid; end
        end
        a = found ? (lo + hi) / 2 : an
        ac[p] = a; valid[p] = found

        # Growth/χ/f and dD/da (central diff) at the crossing; trajectory + dF/da.
        j1, j2, fr = _tabidx(a, logamin, dloga, n, atab)
        e = a * cbrt(eps(T))     # precision-aware central-diff step (≈6e-6 f64, ≈5e-3 f32);
                                 # a fixed 1e-5 catastrophically cancels dDk in Float32
        j1p, j2p, frp = _tabidx(a + e, logamin, dloga, n, atab)
        j1m, j2m, frm = _tabidx(a - e, logamin, dloga, n, atab)
        xs1 = q1; xs2 = q2; xs3 = q3; xd1 = zero(T); xd2 = zero(T); xd3 = zero(T)
        for k in 1:K
            Dkk = Dmat[j1, k] * (1 - fr) + Dmat[j2, k] * fr
            Dp  = Dmat[j1p, k] * (1 - frp) + Dmat[j2p, k] * frp
            Dm  = Dmat[j1m, k] * (1 - frm) + Dmat[j2m, k] * frm
            dD  = (Dp - Dm) / (2e)
            Dk[p, k] = Dkk; dDk[p, k] = dD
            P1k = Psi[p, 1, k]; P2k = Psi[p, 2, k]; P3k = Psi[p, 3, k]
            xs1 += P1k * Dkk; xs2 += P2k * Dkk; xs3 += P3k * Dkk
            xd1 += P1k * dD;  xd2 += P2k * dD;  xd3 += P3k * dD
        end
        chi[p] = chitab[j1] * (1 - fr) + chitab[j2] * fr
        f1[p]  = f1tab[j1] * (1 - fr) + f1tab[j2] * fr
        d1 = xs1 - o1; d2 = xs2 - o2; d3 = xs3 - o3
        r  = sqrt(d1 * d1 + d2 * d2 + d3 * d3)
        drda = (d1 * xd1 + d2 * xd2 + d3 * xd3) / max(r, T(1e-30))
        Ea = sqrt(Om / (a*a*a) + Ok / (a*a) + Ode0 * a^(-3 * (1 + w0 + wa)) * exp(-3 * wa * (1 - a)))
        Fp[p] = drda - (-T(2997.92458) / (a * a * Ea))
    end
end

function _crossing_forward(Psi::AbstractArray{T,3}, q::AbstractMatrix{T},
                           cosmo::Cosmology, obs::NTuple{3,T}, af::T, an::T;
                           n_scan::Int=16, n_bisect::Int=30) where {T}
    N = size(Psi, 1); K = size(Psi, 3)
    atabh = cosmo._a_table; n = length(atabh)
    logamin = log10(T(atabh[1])); dloga = (log10(T(atabh[end])) - logamin) / (n - 1)
    aD   = _devcopy(Psi, T.(atabh))
    Ktab = (cosmo._D1_table, cosmo._D2_table, cosmo._D3a_table, cosmo._D3b_table, cosmo._D3c_table)
    Dmat = _devcopy(Psi, T.(reduce(hcat, Ktab[1:K])))       # (n,K) on device
    chic = _devcopy(Psi, T.(cosmo._chi_table)); f1c = _devcopy(Psi, T.(cosmo._f1_table))
    o1, o2, o3 = obs
    Om = T(Omega_m(cosmo)); Ok = T(cosmo.Omega_k); w0 = T(cosmo.w0); wa = T(cosmo.wa)
    Ode0 = T(1) - Om - Ok

    ac  = similar(Psi, T, N); valid = similar(Psi, Bool, N)
    Dk  = similar(Psi, T, N, K); dDk = similar(Psi, T, N, K)
    chi = similar(Psi, T, N); f1 = similar(Psi, T, N); Fp = similar(Psi, T, N)
    backend = get_backend(Psi)
    _cross_kernel!(backend)(ac, valid, Dk, dDk, chi, f1, Fp, q, Psi, aD, Dmat, chic, f1c,
        K, n, logamin, dloga, af, an, o1, o2, o3, Om, Ok, Ode0, w0, wa,
        n_scan, n_bisect; ndrange=N)
    synchronize(backend)
    return (ac=ac, valid=valid, Dk=Dk, dDk=dDk, chi=chi, f1=f1, Fp=Fp)
end

"""
    lightcone_cross_ad(Psi, q, cosmo, observer, a_far, a_near; rsd=false)
        -> (; x_obs, a_cross, v_r, valid)

Differentiable lightcone crossing (see module docstring).  `Psi::(N,3,K)`, `q::(N,3)`
on any backend; the result is differentiable in Ψ (→ ω).
"""
function lightcone_cross_ad(Psi::AbstractArray{T,3}, q::AbstractMatrix{T},
                            cosmo::Cosmology, observer::AbstractVector,
                            a_far::Real, a_near::Real; rsd::Bool=false, velocity::Bool=false) where {T}
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

    # peculiar-velocity vector v(q)=Σ_k f₁ D_k Ψ_k (comoving Mpc/h; ×aH(a) → km/s). Computed when
    # needed for RSD or when requested for a peculiar-velocity likelihood; differentiable in Ψ (→ ω).
    vvec = if rsd || velocity
        fk = @ignore_derivatives fwd.f1 .* fwd.Dk
        dropdims(sum(reshape(fk, N, 1, K) .* Psi; dims=3); dims=3)
    else
        @ignore_derivatives fill!(similar(x_obs), zero(T))
    end
    v_r = if rsd
        rhat = diff ./ max.(r, T(1e-30))
        vec(sum(vvec .* rhat; dims=2))
    else
        @ignore_derivatives fill!(similar(r, N), zero(T))
    end
    return (x_obs=x_obs, a_cross=a_d, v_r=v_r, v_vec=vvec, valid=fwd.valid)
end
