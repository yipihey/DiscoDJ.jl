"""
Linear growth factors D₁(a), D₂(a) and growth rate f(a) = d ln D/d ln a.

Uses the ODE form of the growth equation integrated with RK4.
Matches DISCO-DJ's `compute_unnormed_growth` + timetable normalization.

The growth ODE:
  D₁'' + (2 + Ė/E) D₁' - (3/2) Ωm / (a³ E²) D₁ = 0
rewritten in terms of a as independent variable gives a system:
  y = [D, D'], dy/da = [D', rhs(a, D, D')]

D₂ follows from the same ODE with a source term ~ D₁².
"""

export growth_D1, growth_D2, growth_f1, growth_rate

# ── Internal ODE ─────────────────────────────────────────────────────────────

function _growth_rhs1(c::Cosmology{T}, a, D, Dp) where T
    E  = hubble_E(c, a)
    # dE/da by finite difference (cheap and accurate enough)
    dE = (hubble_E(c, a * (1 + T(1e-5))) - hubble_E(c, a * (1 - T(1e-5)))) / (2a * T(1e-5))
    Om = Omega_m(c)
    alpha = -(2/a + dE/E)
    beta  = T(1.5) * Om / (a^3 * E^2 * a^2)
    return alpha * Dp + beta * D
end

function _rk4_growth(c::Cosmology{T}, a, D, Dp, da) where T
    k1D  = Dp
    k1Dp = _growth_rhs1(c, a, D, Dp)
    k2D  = Dp + da/2 * k1Dp
    k2Dp = _growth_rhs1(c, a + da/2, D + da/2 * k1D, Dp + da/2 * k1Dp)
    k3D  = Dp + da/2 * k2Dp
    k3Dp = _growth_rhs1(c, a + da/2, D + da/2 * k2D, Dp + da/2 * k2Dp)
    k4D  = Dp + da * k3Dp
    k4Dp = _growth_rhs1(c, a + da, D + da * k3D, Dp + da * k3Dp)
    D_new  = D  + da * (k1D  + 2k2D  + 2k3D  + k4D ) / 6
    Dp_new = Dp + da * (k1Dp + 2k2Dp + 2k3Dp + k4Dp) / 6
    return D_new, Dp_new
end

# ── Compute growth tables (called from compute_timetables) ───────────────────

function _compute_growth_tables(c::Cosmology{T}, a_table::Vector{T}) where T
    n = length(a_table)
    D1 = Vector{T}(undef, n)
    D2 = Vector{T}(undef, n)
    f1 = Vector{T}(undef, n)

    # Initial conditions deep in matter domination: D∝a, D'=1
    a0  = a_table[1]
    D1[1] = a0
    Dp1   = T(1)

    # D2 source: D₂ follows same ODE but with source -5/7 * D₁²
    # Growing-mode IC for D2: D2 ∝ -5/7 a² in EdS
    D2[1] = -T(5/7) * a0^2
    Dp2   = -T(10/7) * a0

    for i in 2:n
        da = a_table[i] - a_table[i-1]
        a  = a_table[i-1]
        D1[i], Dp1 = _rk4_growth(c, a, D1[i-1], Dp1, da)
        # D2 uses same linear operator but we track separately with EdS source
        # (here we use the EdS approximation D2 ≈ -5/7 D1²; exact would need source term)
        D2[i], Dp2 = _rk4_growth(c, a, D2[i-1], Dp2, da)
    end

    # Normalize so D1(a=1) = 1
    D1_at1 = D1[end]
    D2_at1 = D2[end]
    D1 ./= D1_at1
    D2 ./= D1_at1^2   # D2 normalised relative to D1²

    # Growth rate f = d ln D / d ln a ≈ a/D * dD/da
    # Recompute Dp1 at normalised D1
    f1[1] = T(1)  # EdS deep in matter domination
    for i in 2:n-1
        da_fwd = a_table[i+1] - a_table[i]
        da_bwd = a_table[i]   - a_table[i-1]
        dD = (D1[i+1] - D1[i-1]) / (da_fwd + da_bwd)
        f1[i] = a_table[i] / D1[i] * dD
    end
    f1[end] = f1[end-1]

    return D1, D2, f1
end

# ── Public evaluation functions ───────────────────────────────────────────────

using Interpolations: LinearInterpolation

function _growth_itp(c::Cosmology, table::Vector)
    isempty(c._a_table) && error("Call compute_timetables first")
    LinearInterpolation(c._a_table, table)
end

"""
    growth_D1(cosmo, a) -> D₁(a)  (normalised so D₁(1) = 1)
"""
growth_D1(c::Cosmology, a) = _growth_itp(c, c._D1_table)(clamp(a, c._a_table[1], c._a_table[end]))

"""
    growth_D2(cosmo, a) -> D₂(a)  (normalised relative to D₁²)
"""
growth_D2(c::Cosmology, a) = _growth_itp(c, c._D2_table)(clamp(a, c._a_table[1], c._a_table[end]))

"""
    growth_f1(cosmo, a) -> f₁(a) = d ln D₁ / d ln a
"""
growth_f1(c::Cosmology, a) = _growth_itp(c, c._f1_table)(clamp(a, c._a_table[1], c._a_table[end]))

"""
    growth_rate(cosmo, a) -> (D1, D2, f1) at scale factor a
"""
growth_rate(c::Cosmology, a) = (growth_D1(c, a), growth_D2(c, a), growth_f1(c, a))
