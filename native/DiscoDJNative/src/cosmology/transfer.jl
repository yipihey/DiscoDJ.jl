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
    Om_m  = Omega_m(c)
    Om_b  = c.Omega_b
    h     = c.h
    ombh2 = Om_b  * h^2
    ommh2 = Om_m  * h^2
    Θ     = Tcmb / 2.7    # CMB temperature ratio

    # Redshift of equality and drag epoch
    z_eq  = 2.5e4 * ommh2 * Θ^(-4)
    k_eq  = 7.46e-2 * ommh2 * Θ^(-2)   # h/Mpc
    b1    = 0.313 * ommh2^(-0.419) * (1 + 0.607 * ommh2^0.674)
    b2    = 0.238 * ommh2^0.223
    z_d   = 1291 * ommh2^0.251 / (1 + 0.659 * ommh2^0.828) * (1 + b1 * ombh2^b2)

    # Sound horizon at drag epoch
    R_eq  = 31.5e3 * ombh2 * Θ^(-4) * (1000/z_eq)
    R_d   = 31.5e3 * ombh2 * Θ^(-4) * (1000/z_d)
    s     = 2/(3*k_eq) * sqrt(6/R_eq) * log((sqrt(1+R_d) + sqrt(R_d + R_eq)) / (1 + sqrt(R_eq)))

    k_silk = 1.6 * ombh2^0.52 * ommh2^0.01 * (1 + (5.2*ommh2)^(-0.62))^(-1/4)  # not standard, using common approx
    # Silk damping k: Eq 15
    k_silk = 1.6 * (ombh2^0.52) * (ommh2^0.38) * (1 + (5.2 * ommh2)^(-0.62))^(-1/4)

    f_baryon = Om_b / Om_m
    f_cdm    = 1 - f_baryon

    T_arr = similar(k, Float64)
    @inbounds for idx in eachindex(k)
        kh = k[idx]   # k in h/Mpc

        # CDM transfer function
        q = kh / (13.41 * k_eq)
        C0 = 14.2 + 386/(1 + 69.9*q^1.08)
        T0 = log(exp(1) + 1.8*q) / (log(exp(1) + 1.8*q) + C0 * q^2)

        # Baryon transfer function
        y = z_eq / (1 + z_d)
        G = y * (-6*sqrt(1+y) + (2+3y)*log((sqrt(1+y)+1)/(sqrt(1+y)-1)))
        alpha_b = 2.07 * k_eq * s * (1+R_d)^(-3/4) * G
        beta_b  = 0.5 + f_baryon + (3 - 2*f_baryon) * sqrt((17.2*ommh2)^2 + 1)
        beta_node = 8.41 * ommh2^0.435

        s_tilde = s / (1 + (beta_node / (kh*s))^3)^(1/3)
        j0_ks   = sinc(kh * s_tilde / π)   # Julia sinc is normalised

        C_b     = 14.2/alpha_b + 386/(1 + 69.9*q^1.08)
        T_b_tilde = log(exp(1) + 1.8*alpha_b*q) / (log(exp(1) + 1.8*alpha_b*q) + C_b * q^2)
        T_b  = (T_b_tilde / (1 + (kh*s/5.2)^2) + alpha_b/(1 + (beta_b/(kh*s))^3) * exp(-(kh/k_silk)^1.4)) * j0_ks

        T_arr[idx] = f_baryon * T_b + f_cdm * T0
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

    # Primordial spectrum: P_prim(k) ∝ k^n_s (Harrison-Zel'dovich-Peebles)
    # Use pivot k₀ = 0.05 Mpc⁻¹; we work in h/Mpc so k₀ → 0.05/h
    k0 = 0.05 / c.h
    Pk = @. k^(c.n_s) * Tk^2   # proportional; normalise via σ₈

    # σ₈ normalisation via spherical top-hat window integral
    # W(x) = 3(sin x - x cos x)/x³, R8 = 8 Mpc/h
    R8 = 8.0  # Mpc/h
    kR = k .* R8
    W  = @. 3(sin(kR) - kR*cos(kR)) / kR^3
    W[kR .< 1e-3] .= 1.0

    # Integrate sigma8² = (1/2π²) ∫ k² P(k) W²(kR8) dk
    dk = diff(k)
    integrand = @. k^2 * Pk * W^2 / (2π^2)
    sigma8_unnorm = sqrt(sum(0.5*(integrand[1:end-1] .+ integrand[2:end]) .* dk))
    A = (c.sigma8 / sigma8_unnorm)^2
    Pk .*= A

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
