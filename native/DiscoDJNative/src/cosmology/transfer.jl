"""
Transfer functions T(k) for the linear matter power spectrum.

Implements:
- Eisenstein & Hu (1998) with baryon oscillations  [default]
- BBKS (Bardeen, Bond, Kaiser, Szalay 1986)

Usage:
    pk_table = linear_power_spectrum(cosmo, k_vec; transfer="Eisenstein-Hu")

Returns Dict("k" => k, "Pk" => Pk) matching DISCO-DJ's pk_table convention.
"""

export eisenstein_hu, bbks, linear_power_spectrum

# ── Eisenstein & Hu (1998) ───────────────────────────────────────────────────

"""
    eisenstein_hu(cosmo, k; Tcmb=2.72548) -> T(k)

Eisenstein & Hu (1998) fitting formula including baryon acoustic effects.
k in h/Mpc.  Returns the transfer function T(k) (dimensionless, T→1 as k→0).
"""
function eisenstein_hu(c::Cosmology{CT}, k::AbstractArray{T}; Tcmb::Float64=2.72548) where {CT, T}
    # Ported line-for-line from DISCO-DJ's eisenstein_hu (full EH98 with baryons,
    # arXiv:astro-ph/9709112) so the Julia transfer matches the JAX reference.
    θ   = Tcmb / 2.7
    θ2  = θ^2
    θ4  = θ2^2
    omh2 = Omega_m(c) * c.h^2            # Ω0 h²
    f_b  = c.Omega_b / Omega_m(c)        # baryon fraction
    obh2 = omh2 * f_b                    # Ωb h²

    z_eq = 2.50e4 * omh2 / θ4                 # Eq. 2
    k_eq = 0.0746 * omh2 / θ2 / c.h           # Eq. 3 (h/Mpc) — NOTE the /h

    z_d1 = 0.313 * omh2^(-0.419) * (1 + 0.607*omh2^0.674)
    z_d2 = 0.238 * omh2^0.223
    z_d  = 1291.0 * omh2^0.251 / (1 + 0.659*omh2^0.828) * (1 + z_d1*obh2^z_d2)   # Eq. 4

    R_d  = 31.5 * obh2 / θ4 * (1000/(1 + z_d))    # Eq. 5
    R_eq = 31.5 * obh2 / θ4 * (1000/z_eq)         # Eq. 5
    s    = 2/3/k_eq * sqrt(6/R_eq) *
           log((sqrt(1+R_d) + sqrt(R_d+R_eq)) / (1 + sqrt(R_eq)))                # Eq. 6
    k_silk = 1.6 * obh2^0.52 * omh2^0.73 * (1 + (10.4*omh2)^(-0.95)) / c.h       # Eq. 7

    a1 = (46.9*omh2)^0.670 * (1 + (32.1*omh2)^(-0.532))                          # Eq. 11
    a2 = (12.0*omh2)^0.424 * (1 + (45.0*omh2)^(-0.582))
    α_c = a1^(-f_b) * a2^(-f_b^3)
    b1 = 0.944 / (1 + (458*omh2)^(-0.708))                                       # Eq. 12
    b2 = (0.395*omh2)^(-0.0266)
    β_c = 1 / (1 + b1*((1 - f_b)^b2 - 1))

    yG = (1 + z_eq) / (1 + z_d)
    Gy = yG * (-6*sqrt(1+yG) + (2 + 3yG)*log((sqrt(1+yG)+1)/(sqrt(1+yG)-1)))
    α_b = 2.07 * k_eq * s * (1+R_d)^(-0.75) * Gy
    β_b = 0.5 + f_b + (3 - 2*f_b)*sqrt((17.2*omh2)^2 + 1)                        # Eq. 24
    β_node = 8.41 * omh2^0.435                                                   # Eq. 23

    T_arr = similar(k, Float64)
    @inbounds for idx in eachindex(k)
        kk = Float64(k[idx])             # k in h/Mpc
        q  = kk / (13.41 * k_eq)         # Eq. 10
        ks = kk * s
        Cf(ac)     = 14.2/ac + 386/(1 + 69.9*q^1.08)                # Eq. 20
        lt(b)      = log(exp(1) + 1.8*b*q)                          # Eq. 19
        T0t(ac,bc) = lt(bc) / (lt(bc) + Cf(ac)*q^2)                 # Eq. 19
        f   = 1 / (1 + (ks/5.4)^4)                                  # Eq. 18
        T_c = f*T0t(1.0, β_c) + (1-f)*T0t(α_c, β_c)                 # Eq. 17
        s_tilde = s / (1 + (β_node/ks)^3)^(1/3)                     # Eq. 22
        Tb1 = T0t(1.0, 1.0) / (1 + (ks/5.2)^2)
        Tb2 = α_b / (1 + (β_b/ks)^3) * exp(-(kk/k_silk)^1.4)
        T_b = sinc(kk*s_tilde/π) * (Tb1 + Tb2)                      # Eq. 21 (sinc normalised)
        T_arr[idx] = f_b*T_b + (1 - f_b)*T_c                        # Eq. 8
    end
    return T_arr
end

# ── BBKS (Bardeen, Bond, Kaiser, Szalay 1986) ────────────────────────────────

"""
    bbks(cosmo, k; Neff=3.046) -> T(k)

BBKS fitting formula. k in h/Mpc.
"""
function bbks(c::Cosmology{CT}, k::AbstractArray{T}; Neff::Float64=3.046) where {CT, T}
    Om_m = Omega_m(c)
    h    = c.h
    Γ    = Om_m * h * exp(-c.Omega_b * (1 + sqrt(2h)/Om_m))   # shape parameter
    T_arr = similar(k, Float64)
    @inbounds for idx in eachindex(k)
        q = k[idx] / Γ   # k in h/Mpc, q in Mpc
        T_arr[idx] = log(1 + 2.34q) / (2.34q) *
                     (1 + 3.89q + (16.1q)^2 + (5.46q)^3 + (6.71q)^4)^(-1/4)
    end
    return T_arr
end

# ── Linear P(k) table ────────────────────────────────────────────────────────

const TRANSFER_REGISTRY = Dict{String, Function}(
    "Eisenstein-Hu" => (c, k) -> eisenstein_hu(c, k),
    "EH"            => (c, k) -> eisenstein_hu(c, k),
    "BBKS"          => (c, k) -> bbks(c, k),
)

"""
    linear_power_spectrum(cosmo, k; transfer="Eisenstein-Hu", n_modes=256) -> Dict

Compute the linear matter power spectrum P(k) = A_s * k^ns * T²(k) * (k/k_pivot)^(ns-1).
Returns Dict("k" => k_vec, "Pk" => Pk_vec) matching DISCO-DJ's pk_table convention.

σ₈ normalisation is applied if `cosmo.sigma8 > 0`.

k in h/Mpc, P(k) in (Mpc/h)³.
"""
function linear_power_spectrum(c::Cosmology{CT}, k::AbstractArray;
                                transfer::String="Eisenstein-Hu",
                                n_modes::Int=256) where CT
    haskey(TRANSFER_REGISTRY, transfer) || error("Unknown transfer function: $transfer")
    Tk = TRANSFER_REGISTRY[transfer](c, k)

    # Primordial spectrum P_prim(k) = k^n_s (DISCO-DJ's compute_primordial_ps,
    # normalization=1), then × T²(k); σ₈-normalised below (as JAX does for EH).
    Pk = @. k^(c.n_s) * Tk^2

    # σ₈ from P(k) exactly as DISCO-DJ's get_sigma8_squared_from_Pk:
    #   σ8² = (1/2π²) ∫ k³ P(k) W²(kR) d(ln k),  R = 8 Mpc/h,  trapezoid in ln k.
    R8 = 8.0
    x  = k .* R8
    W  = @. 3 * (sin(x) - x*cos(x)) / x^3
    integrand = @. k^3 * Pk * W^2
    logk = log.(k)
    sigma8_sq = sum(0.5 .* (integrand[1:end-1] .+ integrand[2:end]) .* diff(logk)) / (2π^2)
    Pk = Pk .* (c.sigma8^2 / sigma8_sq)

    return Dict("k" => collect(Float64, k), "Pk" => collect(Float64, Pk))
end

"""
    linear_power_spectrum(cosmo; kmin=1e-4, kmax=50, n_modes=512, transfer="Eisenstein-Hu")

Convenience method: auto-generates a log-spaced k grid.
"""
function linear_power_spectrum(c::Cosmology; kmin=1e-4, kmax=50.0,
                                n_modes::Int=512, transfer::String="Eisenstein-Hu")
    k = exp10.(LinRange(log10(kmin), log10(kmax), n_modes))
    return linear_power_spectrum(c, k; transfer, n_modes)
end
