"""
Evaluate LPT fields at a given scale factor a.

Functions here combine the growth-factor-weighted displacement fields to produce:
- `evaluate_lpt_psi_at_a`      → displacement ψ(a)   [Mpc/h], shape (res³,3)
- `evaluate_lpt_pos_at_a`      → positions x(a)       [Mpc/h], periodic [0,L)
- `evaluate_lpt_psi_dot_at_a`  → conjugate velocity ψ̇ [Mpc/h], conformal time units

Matches the Python DISCO-DJ API exactly.
"""

export evaluate_lpt_psi_at_a, evaluate_lpt_pos_at_a, evaluate_lpt_psi_dot_at_a,
       evaluate_lpt_eulerian_acc_at_a

# ── Lagrangian grid ───────────────────────────────────────────────────────────

"""
    lagrangian_grid(res, boxsize; T=Float64) -> Array{T}(res^3, 3)

Uniform Lagrangian particle positions on a Cartesian grid.
Returns q in [0, boxsize) (open interval, so q = i·Δx for i=0…res-1).
"""
function lagrangian_grid(res::Int, boxsize::Real; T::Type{<:AbstractFloat}=Float64)
    dx  = T(boxsize) / res
    ax  = collect(T(0):dx:T(boxsize) - dx)
    grid = Array{T}(undef, res^3, 3)
    idx  = 1
    for ix in 1:res, iy in 1:res, iz in 1:res
        grid[idx, 1] = ax[ix]
        grid[idx, 2] = ax[iy]
        grid[idx, 3] = ax[iz]
        idx += 1
    end
    return grid
end

# ── Growth-weighted displacement ───────────────────────────────────────────────

# Displacement fields may be stored f32 (Array/CuArray) or packed f16 (HalfField);
# `_expand` returns an f32 array either way (a no-op in the f32 case).
_expand(x::AbstractArray) = x
_expand(x::HalfField)     = expand_half(x)

function _psi_at_a(lpt::LPTResult{T}, cosmo::Cosmology, a::Real) where T
    D1 = T(growth_D1(cosmo, a))
    psi = D1 .* _expand(lpt.psi1)

    if lpt.n_order >= 2 && lpt.psi2 !== nothing
        # `compute_lpt`'s ψ₂ already carries the EdS -3/7 spatial coefficient, so
        # its time growth is D₁² (the faithful exact-growth path uses D₂plus·ψ₂ₑₓ
        # instead — see `evaluate_core` in nlpt_core.jl).
        D2 = D1^2
        psi .+= D2 .* _expand(lpt.psi2)
    end

    if lpt.n_order >= 3 && lpt.psi3 !== nothing
        D3 = T(growth_D1(cosmo, a))^3   # EdS approximation for 3LPT
        psi .+= D3 .* _expand(lpt.psi3) .* T(1/3)
    end
    return psi
end

"""
    evaluate_lpt_psi_at_a(lpt, cosmo, a; n_order=nothing) -> Array (res,res,res,3)

Displacement field ψ(a) = Σ_n D_n(a)·ψ_n in Mpc/h.
"""
function evaluate_lpt_psi_at_a(lpt::LPTResult{T}, cosmo::Cosmology, a::Real;
                                n_order::Union{Nothing,Int}=nothing) where T
    effective = something(n_order, lpt.n_order)
    effective <= lpt.n_order || error("Requested n_order=$effective > computed n_order=$(lpt.n_order)")
    # Temporarily cap the lpt.n_order by returning a view with reduced order
    lpt_eff = LPTResult{T}(lpt.psi1,
                           effective >= 2 ? lpt.psi2 : nothing,
                           effective >= 3 ? lpt.psi3 : nothing,
                           effective, lpt.res, lpt.boxsize)
    return _psi_at_a(lpt_eff, cosmo, a)
end

"""
    evaluate_lpt_pos_at_a(lpt, cosmo, a; n_order=nothing) -> Array (res^3, 3)

Eulerian particle positions x(a) = q + ψ(a), periodic in [0, L).
Returns flat (N_particles, 3) array.
"""
function evaluate_lpt_pos_at_a(lpt::LPTResult{T}, cosmo::Cosmology, a::Real;
                                n_order::Union{Nothing,Int}=nothing) where T
    psi = evaluate_lpt_psi_at_a(lpt, cosmo, a; n_order)
    L   = T(lpt.boxsize)
    res = lpt.res
    q   = lagrangian_grid(res, L; T)
    # Flatten ψ to (N,3)
    psi_flat = reshape(psi, res^3, 3)
    pos = q .+ psi_flat
    # Periodic wrap
    @inbounds for i in eachindex(pos)
        pos[i] = mod(pos[i], L)
    end
    return pos
end

"""
    evaluate_lpt_psi_dot_at_a(lpt, cosmo, a; n_order=nothing) -> Array (res,res,res,3)

Conjugate velocity ψ̇(a) = d(D·ψ)/d(conformal time).
In Gadget convention (√a-scaled) this is:
    ψ̇ = H(a)·f₁(a)·D₁·ψ₁ + …  ×  (a² E(a))  [Mpc/h × km/s / (Mpc/h)]

Here we return the displacement-rate derivative with respect to conformal time τ,
so that the velocity in peculiar km/s is v_pec = ψ̇ / a.
"""
function evaluate_lpt_psi_dot_at_a(lpt::LPTResult{T}, cosmo::Cosmology, a::Real;
                                    n_order::Union{Nothing,Int}=nothing) where T
    effective = something(n_order, lpt.n_order)
    H0 = T(100 * cosmo.h)   # km/s/(Mpc/h)
    E  = T(hubble_E(cosmo, a))
    f1 = T(growth_f1(cosmo, a))
    D1 = T(growth_D1(cosmo, a))

    vel = f1 * D1 * E * H0 * _expand(lpt.psi1)

    if effective >= 2 && lpt.psi2 !== nothing
        # f₂ ≈ 2f₁ in EdS; ψ₂ carries the -3/7 coefficient so its growth is D₁²
        f2 = T(2) * f1
        D2 = D1^2
        vel .+= f2 * D2 * E * H0 * _expand(lpt.psi2)
    end
    return vel
end

"""
    evaluate_lpt_eulerian_acc_at_a(lpt, cosmo, a) -> Array (res,res,res,3)

Lagrangian acceleration at Eulerian positions: acceleration in Mpc/h/(km/s)².
(Used for COLA-style corrections; matches Python's `evaluate_lpt_eulerian_acc_at_a`.)
"""
function evaluate_lpt_eulerian_acc_at_a(lpt::LPTResult{T}, cosmo::Cosmology, a::Real) where T
    H0 = T(100 * cosmo.h)
    E  = T(hubble_E(cosmo, a))
    f1 = T(growth_f1(cosmo, a))
    D1 = T(growth_D1(cosmo, a))
    # Eulerian acceleration = dΨ̇/dτ + H·ψ̇  (in conformal time)
    # Simplified: 2nd conformal-time derivative of D(a)·ψ₁
    H  = E * H0
    dH_da = (hubble_E(cosmo, a * (1+1e-4)) - hubble_E(cosmo, a * (1-1e-4))) / (2a*1e-4) * H0
    p1 = _expand(lpt.psi1)
    acc = @. -T(1.5) * T(Omega_m(cosmo)) * H0^2 / a^3 * D1 * p1
    return acc
end
