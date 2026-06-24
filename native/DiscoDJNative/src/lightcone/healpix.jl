"""
Pure-Julia HEALPix utilities.

Implements:
- `ang2pix_ring`    — (θ,φ) → pixel index in RING scheme
- `accumulate_map`  — deposit weighted particles onto a HEALPix map
- `shells_to_overdensity` — convert shell counts to overdensity δ
- `density_shells_to_kappa` — Born-approximation weak-lensing convergence κ

Matches DISCO-DJ's `discodj.core.healpix` module (validated against healpy).
No external HEALPix dependency — pure Julia arithmetic.
"""

export ang2pix_ring, accumulate_map, shells_to_overdensity, density_shells_to_kappa,
       MapSpec

using LinearAlgebra

# ── ang2pix_ring ──────────────────────────────────────────────────────────────

"""
    ang2pix_ring(nside, theta, phi) -> Int (0-based pixel index)

Convert colatitude θ ∈ [0,π] and longitude φ ∈ [0,2π) to HEALPix RING pixel.
Validated to match healpy.ang2pix(nside, theta, phi, nest=False).
"""
function ang2pix_ring(nside::Int, theta::Real, phi::Real)
    nside >= 1 || throw(ArgumentError("nside must be ≥ 1"))
    z   = cos(theta)
    phi = mod(phi, 2π)

    tp  = phi / (π/2)   # phi in [0,4)
    za  = abs(z)

    if za <= 2/3   # Equatorial region
        temp1 = nside * (0.5 + tp)
        temp2 = nside * z * 0.75
        jp = floor(Int, temp1 - temp2) + 1   # index of ascending edge
        jm = floor(Int, temp1 + temp2) + 1   # index of descending edge

        ir = nside + 1 + jp + jm   # kphi
        kshift = (ir & 1) == 0 ? 1 : 0    # kshift=1 if ir even

        ip = floor(Int, (jp + jm - nside + 1 + kshift) / 2)
        ip = mod(ip, 4*nside)

        ipix = 2*nside*(nside-1) + (4*nside)*(ir-1) + ip
    else    # Polar region
        tp  = phi / (π/2)
        ntt = floor(Int, tp)
        tp  = tp - ntt

        if za > 2/3
            # North
            jp = floor(Int, nside * sqrt(3*(1-za)) * tp)     + 1
            jm = floor(Int, nside * sqrt(3*(1-za)) * (1-tp)) + 1
        else
            # South
            jp = floor(Int, nside * sqrt(3*(1+za)) * tp)     + 1
            jm = floor(Int, nside * sqrt(3*(1+za)) * (1-tp)) + 1
        end

        ir  = min(jp, jm)
        ir  = min(ir, nside)
        kshift = (z >= 0) ? 0 : 1
        ip  = floor(Int, min(tp, 1-tp) * 4 * ir - kshift/2) + 1
        ip  = mod(ip-1, 4*ir) + 1

        if z > 2/3
            ipix = 2*ir*(ir-1) + ip - 1
        else
            ipix = 12*nside^2 - 2*ir*(ir+1) + ip - 1
        end
    end
    return ipix
end

# ── accumulate_map ────────────────────────────────────────────────────────────

"""
    accumulate_map(nside, theta, phi, weights=nothing) -> Vector{Float64}

Deposit particles at (θ,φ) onto a HEALPix RING map of given nside.
Returns a (12*nside²,) map of counts or weighted sums.
"""
function accumulate_map(nside::Int,
                        theta::AbstractVector{T}, phi::AbstractVector{T},
                        weights::Union{AbstractVector,Nothing}=nothing) where T
    npix = 12 * nside^2
    hmap = zeros(Float64, npix)
    N    = length(theta)
    @inbounds for i in 1:N
        ipix = ang2pix_ring(nside, theta[i], phi[i]) + 1   # 1-based
        w    = weights === nothing ? 1.0 : Float64(weights[i])
        hmap[ipix] += w
    end
    return hmap
end

# ── MapSpec ───────────────────────────────────────────────────────────────────

"""
    MapSpec(; nside, a_edges, weighted=true)

Configuration for accumulating shell maps during lightcone generation.
- `nside`    — HEALPix resolution parameter
- `a_edges`  — shell edges in scale factor (length n_shells+1)
- `weighted` — weight particles by mass
"""
Base.@kwdef struct MapSpec
    nside::Int
    a_edges::Vector{Float64}
    weighted::Bool = true
end

n_shells(ms::MapSpec) = length(ms.a_edges) - 1
n_pix(ms::MapSpec)    = 12 * ms.nside^2

# ── shells_to_overdensity ─────────────────────────────────────────────────────

"""
    shells_to_overdensity(counts) -> Array

Convert raw shell count maps to overdensity δ = n/n̄ - 1 for each shell.
`counts` shape: (n_shells, n_pix).
"""
function shells_to_overdensity(counts::AbstractMatrix)
    n_shells, n_pix = size(counts)
    delta = similar(counts, Float64)
    for k in 1:n_shells
        n_bar = mean(counts[k, :])
        if n_bar > 0
            delta[k, :] = counts[k, :] ./ n_bar .- 1
        else
            delta[k, :] .= 0
        end
    end
    return delta
end

mean(x) = sum(x) / length(x)

# ── density_shells_to_kappa ───────────────────────────────────────────────────

"""
    density_shells_to_kappa(delta, a_edges, cosmo; z_source=1.0) -> Vector{Float64}

Compute the Born-approximation weak-lensing convergence κ by integrating
δ(χ) with the lensing kernel W(χ, χ_s).

W(χ, χ_s) = (3Ω_m H₀²)/(2c²) · χ(1 - χ/χ_s) / a

Returns κ map (n_pix,) for a single source plane at redshift z_source.
`delta` shape: (n_shells, n_pix).
"""
function density_shells_to_kappa(delta::AbstractMatrix{T}, a_edges::AbstractVector,
                                  cosmo::Cosmology; z_source::Real=1.0) where T
    n_shells, n_pix = size(delta)
    kappa = zeros(Float64, n_pix)

    a_s   = 1/(1 + z_source)
    chi_s = comoving_distance(cosmo, a_s)
    Om_m  = Omega_m(cosmo)
    H0    = 100 * cosmo.h  # km/s/Mpc
    c_kms = 299792.458     # km/s
    prefac = 1.5 * Om_m * (H0/c_kms)^2

    for k in 1:n_shells
        a_k    = (a_edges[k] + a_edges[k+1]) / 2
        chi_k  = comoving_distance(cosmo, a_k)
        dchi   = comoving_distance(cosmo, a_edges[k]) - comoving_distance(cosmo, a_edges[k+1])
        dchi   = abs(dchi)

        W_k = chi_k <= chi_s ?
              prefac * chi_k * (1 - chi_k/chi_s) / a_k :
              0.0

        kappa .+= W_k * delta[k, :] * dchi
    end
    return kappa
end
