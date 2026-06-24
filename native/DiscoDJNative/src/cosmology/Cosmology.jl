"""
Background cosmology: Hubble function, comoving distance, timetables.

Supports flat ΛCDM + w0/wa dark energy and curvature via E(a)² sum.
Parameters match the Python DISCO-DJ Cosmology class exactly.
"""

export Cosmology, PREDEFINED_COSMOLOGIES
export hubble_E, hubble_H, comoving_distance, scale_factor_from_chi
export compute_timetables, growth_rate, Omega_m

using Interpolations: LinearInterpolation

# ── Struct ───────────────────────────────────────────────────────────────────

"""
    Cosmology(; Omega_c, Omega_b, h, sigma8, n_s, Omega_k=0, w0=-1, wa=0)

Eight-parameter cosmology matching DISCO-DJ's Cosmology class.
After construction call `compute_timetables` to enable chi/a interpolation.
"""
Base.@kwdef struct Cosmology{T<:AbstractFloat}
    Omega_c::T  = T(0.2589)
    Omega_b::T  = T(0.0486)
    h::T        = T(0.6774)
    sigma8::T   = T(0.8159)
    n_s::T      = T(0.9667)
    Omega_k::T  = T(0.0)
    w0::T       = T(-1.0)
    wa::T       = T(0.0)
    # timetable fields (populated by compute_timetables)
    _a_table::Vector{T}   = T[]
    _chi_table::Vector{T} = T[]
    _D1_table::Vector{T}  = T[]
    _D2_table::Vector{T}  = T[]
    _f1_table::Vector{T}  = T[]
end

Omega_m(c::Cosmology) = c.Omega_c + c.Omega_b
Omega_de(c::Cosmology, a) = (1 - Omega_m(c) - c.Omega_k) * a^(-3*(1 + c.w0 + c.wa)) * exp(-3*c.wa*(1 - a))

"""
    hubble_E(cosmo, a) -> dimensionless H/H0

E²(a) = Ωm a⁻³ + Ωk a⁻² + Ωde(a)
"""
function hubble_E(c::Cosmology{T}, a) where T
    Om = Omega_m(c)
    sqrt(Om * a^(-3) + c.Omega_k * a^(-2) + Omega_de(c, a))
end

"""
    hubble_H(cosmo, a) -> H(a) in km/s/Mpc
"""
hubble_H(c::Cosmology, a) = 100 * c.h * hubble_E(c, a)

# ── Comoving distance (chi) via RK4 quadrature ───────────────────────────────

function _dchi_da(c::Cosmology{T}, a) where T
    # dχ/da = c / (a² H(a)) = 1 / (a² E(a)) in units of c/H0
    # We store chi in Mpc/h so multiply by c/H0 = 2997.92 Mpc/h
    return T(2997.92458) / (a^2 * hubble_E(c, a))
end

function _rk4_step(f, y, x, dx)
    k1 = f(x)
    k2 = f(x + dx/2)
    k3 = f(x + dx/2)
    k4 = f(x + dx)
    return y + dx * (k1 + 2k2 + 2k3 + k4) / 6
end

"""
    compute_timetables(cosmo; n_pts=1000, a_ini=1e-4) -> Cosmology

Precompute chi(a) on a grid and return a new Cosmology with interpolation tables.
"""
function compute_timetables(c::Cosmology{T}; n_pts::Int=1000, a_ini::T=T(1e-4)) where T
    a_table = collect(LinRange(a_ini, T(1.0), n_pts))
    # Forward integration: chi_fwd[i] = ∫_{a_ini}^{a[i]} dchi/da da
    chi_fwd = Vector{T}(undef, n_pts)
    chi_fwd[1] = T(0)
    da = (T(1) - a_ini) / (n_pts - 1)
    for i in 2:n_pts
        a = a_table[i-1]
        k1 = _dchi_da(c, a)
        k2 = _dchi_da(c, a + da/2)
        k3 = _dchi_da(c, a + da/2)
        k4 = _dchi_da(c, a + da)
        chi_fwd[i] = chi_fwd[i-1] + da * (k1 + 2k2 + 2k3 + k4) / 6
    end
    # Physical chi(a) = chi_total - chi_fwd(a): chi(a=1)=0, chi(a_ini)=max
    chi_table = chi_fwd[end] .- chi_fwd
    # Growth factors (populated by growth.jl after include)
    D1, D2, f1 = _compute_growth_tables(c, a_table)
    return Cosmology{T}(
        Omega_c = c.Omega_c, Omega_b = c.Omega_b, h = c.h,
        sigma8 = c.sigma8, n_s = c.n_s, Omega_k = c.Omega_k,
        w0 = c.w0, wa = c.wa,
        _a_table = a_table, _chi_table = chi_table,
        _D1_table = D1, _D2_table = D2, _f1_table = f1,
    )
end

function _chi_interp(c::Cosmology{T}, a) where T
    isempty(c._a_table) && error("Call compute_timetables first")
    itp = LinearInterpolation(c._a_table, c._chi_table)
    itp(clamp(a, c._a_table[1], c._a_table[end]))
end

"""
    comoving_distance(cosmo, a) -> chi [Mpc/h]
"""
comoving_distance(c::Cosmology, a) = _chi_interp(c, a)

"""
    scale_factor_from_chi(cosmo, chi) -> a

Inverse of chi(a) via bisection on the timetable.
"""
function scale_factor_from_chi(c::Cosmology{T}, chi::T) where T
    isempty(c._a_table) && error("Call compute_timetables first")
    chi_max = c._chi_table[1]   # chi_table[1] = chi(a_ini) = maximum distance
    chi <= 0 && return T(1)
    chi >= chi_max && return c._a_table[1]
    # chi_table is monotonically decreasing → reverse for increasing knots
    itp = LinearInterpolation(reverse(c._chi_table), reverse(c._a_table))
    itp(clamp(chi, T(0), chi_max))
end

# ── Predefined cosmologies ───────────────────────────────────────────────────

const PREDEFINED_COSMOLOGIES = Dict{String, NamedTuple}(
    "Planck18EEBAOSN" => (Omega_c=0.2589, Omega_b=0.0486, h=0.6774, sigma8=0.8159, n_s=0.9667, Omega_k=0.0, w0=-1.0, wa=0.0),
    "Planck15"        => (Omega_c=0.2589, Omega_b=0.0486, h=0.6774, sigma8=0.8159, n_s=0.9667, Omega_k=0.0, w0=-1.0, wa=0.0),
    "Quijote"         => (Omega_c=0.3175, Omega_b=0.049,  h=0.6711, sigma8=0.834,  n_s=0.9624, Omega_k=0.0, w0=-1.0, wa=0.0),
    "CamelsCV"        => (Omega_c=0.2514, Omega_b=0.049,  h=0.6711, sigma8=0.818,  n_s=0.9624, Omega_k=0.0, w0=-1.0, wa=0.0),
)

"""
    Cosmology(name::String) -> Cosmology{Float64}

Construct a predefined cosmology by name. Calls `compute_timetables` automatically.
Valid names: "Planck18EEBAOSN", "Planck15", "Quijote", "CamelsCV".
"""
function Cosmology(name::String; T::Type{<:AbstractFloat}=Float64)
    haskey(PREDEFINED_COSMOLOGIES, name) || error("Unknown cosmology: $name. Available: $(keys(PREDEFINED_COSMOLOGIES))")
    p = PREDEFINED_COSMOLOGIES[name]
    c = Cosmology{T}(; Omega_c=T(p.Omega_c), Omega_b=T(p.Omega_b), h=T(p.h),
                     sigma8=T(p.sigma8), n_s=T(p.n_s), Omega_k=T(p.Omega_k),
                     w0=T(p.w0), wa=T(p.wa))
    return compute_timetables(c)
end

function Base.show(io::IO, c::Cosmology{T}) where T
    print(io, "Cosmology{$T}(Ωm=$(round(Omega_m(c), digits=4)), h=$(c.h), σ8=$(c.sigma8), ns=$(c.n_s), w0=$(c.w0))")
end
