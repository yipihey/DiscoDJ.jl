"""
Sky coordinate projection and masking.

Converts Cartesian particle positions to (RA, Dec, z_cosmo, z_obs) sky coordinates,
with optional redshift-space distortions (RSD).

Matches DISCO-DJ's `discodj.lpt.sky` module API.
"""

export cartesian_to_sky, sky_mask

using LinearAlgebra

"""
    cartesian_to_sky(x, observer, cosmo; v_radial=nothing, with_rsd=false) -> NamedTuple

Convert (N,3) Cartesian comoving positions to sky coordinates.

Returns named tuple:
- `ra`       — right ascension [degrees, 0–360)
- `dec`      — declination [degrees, -90–+90]
- `z_cosmo`  — cosmological redshift (from chi)
- `z_obs`    — observed redshift (with RSD if `with_rsd=true`)
- `chi`      — comoving distance [Mpc/h]
"""
function cartesian_to_sky(x::AbstractMatrix{T}, observer::AbstractVector,
                          cosmo::Cosmology;
                          v_radial::Union{AbstractVector,Nothing}=nothing,
                          with_rsd::Bool=false) where T
    N   = size(x, 1)
    ra  = Vector{T}(undef, N)
    dec = Vector{T}(undef, N)
    chi = Vector{T}(undef, N)
    z_cosmo = Vector{T}(undef, N)
    z_obs   = Vector{T}(undef, N)

    @inbounds for i in 1:N
        dx = x[i,1] - T(observer[1])
        dy = x[i,2] - T(observer[2])
        dz = x[i,3] - T(observer[3])
        d  = sqrt(dx^2 + dy^2 + dz^2)
        chi[i] = d

        # Spherical coords: RA from x-y plane, Dec from z
        ra[i]  = mod(atan(dy, dx) * T(180/π), T(360))
        dec[i] = asin(clamp(dz / max(d, T(1e-10)), T(-1), T(1))) * T(180/π)

        # Cosmological redshift from chi
        a_cross = scale_factor_from_chi(cosmo, d)
        z_cosmo[i] = T(1)/a_cross - T(1)

        # Observed redshift with RSD
        if with_rsd && v_radial !== nothing
            v_pec = v_radial[i] / sqrt(a_cross)   # Gadget √a → peculiar km/s
            c_kms = T(299792.458)
            z_obs[i] = (1 + z_cosmo[i]) * (1 + v_pec / c_kms) - T(1)
        else
            z_obs[i] = z_cosmo[i]
        end
    end

    return (ra=ra, dec=dec, z_cosmo=z_cosmo, z_obs=z_obs, chi=chi)
end

"""
    sky_mask(ra, dec, z; z_range=nothing, healpix_mask=nothing) -> BitVector

Return a Boolean mask selecting particles within the given sky cuts:
- `z_range = (z_min, z_max)` — redshift interval filter
- `healpix_mask` — Boolean HEALPix map (nside inferred from length); particle
  must land on a True pixel to pass. (Requires `healpix_ang2pix` from healpix.jl.)
"""
function sky_mask(ra::AbstractVector, dec::AbstractVector, z::AbstractVector;
                  z_range::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
                  healpix_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    N = length(ra)
    mask = trues(N)

    if z_range !== nothing
        z_min, z_max = z_range
        @inbounds for i in 1:N
            mask[i] &= (z_min <= z[i] <= z_max)
        end
    end

    if healpix_mask !== nothing
        nside  = round(Int, sqrt(length(healpix_mask)/12))
        @inbounds for i in 1:N
            mask[i] || continue
            theta = deg2rad(90 - dec[i])
            phi   = deg2rad(ra[i])
            ipix  = ang2pix_ring(nside, theta, phi) + 1   # 1-based
            mask[i] &= healpix_mask[ipix]
        end
    end

    return mask
end
