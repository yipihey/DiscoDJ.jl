"""
HDF5 I/O for lightcone catalogues.

Writes and reads the Gadget-compatible HDF5 schema defined in the
DISCO-DJ lightcone integration guide (§2), with Blosc-zstd compression.

On-disk format:
  /Header  — cosmology + geometry attributes (LightconeMode=1)
  /PartType1/Coordinates        (M,3)  float32  comoving Mpc/h
  /PartType1/RadialVelocity     (M,)   float32  Gadget √a-scaled km/s
  /PartType1/Velocities         (M,3)  float32  (optional, v_mode=:full)
  /PartType1/ScaleFactor        (M,)   float32
  /PartType1/ParticleIDs        (M,)   uint64
  /PartType1/Masses             (M,)   float32  10¹⁰ M☉/h
  /PartType1/ReplicaIndex       (M,)   int16
  /PartType1/ShellIndex         (M,)   int16
  /PartType1/LagrangianParticleIndex (M,) int32  (optional)
"""

export write_lightcone_hdf5, read_lightcone_header

using HDF5

# ── Header ────────────────────────────────────────────────────────────────────

function _write_header!(f::HDF5.File, cosmo::Cosmology, observer::AbstractVector,
                        boxsize::Real, n_particles::Int, n_per_replica::Int,
                        n_replicas::Int, has_particle_idx::Bool)
    g = create_group(f, "Header")
    attrs = attributes(g)
    attrs["LightconeMode"]          = Int32(1)
    attrs["Observer"]               = Float64.(observer)
    attrs["BoxSize"]                = Float64(boxsize)
    attrs["Omega0"]                 = Float64(Omega_m(cosmo))
    attrs["OmegaLambda"]            = Float64(1 - Omega_m(cosmo) - cosmo.Omega_k)
    attrs["HubbleParam"]            = Float64(cosmo.h)
    attrs["NumPart_PerReplica"]     = Int64(n_per_replica)
    attrs["NumFilesPerSnapshot"]    = Int32(1)
    attrs["Time"]                   = Float64(1.0)
    attrs["NumReplicas"]            = Int32(n_replicas)
    # Particle counts (split for >2³²)
    n_low  = n_particles & 0xFFFFFFFF
    n_high = n_particles >> 32
    pt = zeros(Int32, 6); pt[2] = Int32(n_low)
    ph = zeros(Int32, 6); ph[2] = Int32(n_high)
    attrs["NumPart_Total"]          = pt
    attrs["NumPart_Total_HighWord"] = ph
    attrs["NumPart_ThisFile"]       = pt
    mt = zeros(Float64, 6)
    attrs["MassTable"]              = mt
    attrs["HasLagrangianParticleIndex"] = Int32(has_particle_idx ? 1 : 0)
end

"""
    read_lightcone_header(path) -> Dict

Read the /Header attributes from a lightcone HDF5 file.
Extra computed keys: `n_particles`, `v_mode`, `has_particle_idx`.
"""
function read_lightcone_header(path::String)
    meta = Dict{String,Any}()
    h5open(path, "r") do f
        g = f["Header"]
        for k in keys(attributes(g))
            meta[k] = read(attributes(g)[k])
        end
    end
    low  = meta["NumPart_Total"][2]
    high = meta["NumPart_Total_HighWord"][2]
    meta["n_particles"]       = Int(low) + (Int(high) << 32)
    meta["has_particle_idx"]  = get(meta, "HasLagrangianParticleIndex", 0) == 1
    return meta
end

# ── Writer ────────────────────────────────────────────────────────────────────

"""
    write_lightcone_hdf5(path, crossings, lpt, cosmo, observer;
                         v_mode=:radial, keep_particle_idx=true,
                         compression=:zstd)

Write a lightcone catalogue to HDF5.

`crossings` — output of `find_lightcone_crossings` or `evaluate_lpt_lightcone`.
`lpt`       — LPTResult (needed for velocity computation).
`cosmo`     — Cosmology with timetables.
`observer`  — (3,) observer position [Mpc/h].
`v_mode`    — `:radial` (scalar, default) or `:full` (3-vector).
"""
function write_lightcone_hdf5(path::String, crossings::NamedTuple,
                               lpt::LPTResult{T}, cosmo::Cosmology,
                               observer::AbstractVector;
                               v_mode::Symbol=:radial,
                               keep_particle_idx::Bool=true,
                               compression::Symbol=:zstd) where T
    # Build compression kwargs
    comp_kwargs = compression == :none ? () : (compress=3,)

    M        = length(crossings.a_cross)
    n_per_rep = lpt.res^3
    n_rep     = maximum(crossings.replica_idx; init=0) + 1

    # Compute particle mass: Ω_m ρ_crit L³ / N [10¹⁰ M☉/h]
    rho_crit = 2.775e11   # h² M☉/Mpc³
    mass_unit = Omega_m(cosmo) * rho_crit * lpt.boxsize^3 / n_per_rep / 1e10
    masses    = fill(Float32(mass_unit), M)

    # Compute velocities
    # v_gadget = dΨ/dτ × (100/a^1.5) in Gadget convention
    psi1_flat = reshape(lpt.psi1, n_per_rep, 3)
    psi2_flat = lpt.psi2 !== nothing ? reshape(lpt.psi2, n_per_rep, 3) : nothing

    v_radial  = Vector{Float32}(undef, M)
    v_full    = v_mode == :full ? Matrix{Float32}(undef, M, 3) : nothing

    for i in 1:M
        pid  = Int(crossings.particle_idx[i]) + 1   # 1-based
        a    = crossings.a_cross[i]
        H0   = 100 * cosmo.h
        E    = hubble_E(cosmo, a)
        f1   = growth_f1(cosmo, a)
        D1   = growth_D1(cosmo, a)
        H    = E * H0

        v_vec = f1 * D1 * H * psi1_flat[pid, :]
        if psi2_flat !== nothing
            f2 = 2 * f1
            D2 = D1^2   # ψ₂ carries EdS -3/7 → growth D₁²
            v_vec .+= f2 * D2 * H .* psi2_flat[pid, :]
        end
        # Gadget convention: v_g = v · (100 / a^1.5)
        v_gadget = v_vec .* (100 / a^1.5)

        # Radial component
        x    = crossings.x[i, :]
        n_hat = (x .- observer) ./ max(norm(x .- observer), 1e-10)
        v_radial[i] = Float32(dot(v_gadget, n_hat))

        if v_mode == :full
            v_full[i, :] = Float32.(v_gadget)
        end
    end

    h5open(path, "w") do f
        _write_header!(f, cosmo, observer, lpt.boxsize, M, n_per_rep, n_rep,
                       keep_particle_idx)
        g = create_group(f, "PartType1")

        # Write datasets
        g["Coordinates"]   = Matrix{Float32}(crossings.x)
        g["ScaleFactor"]   = Vector{Float32}(crossings.a_cross)
        g["ParticleIDs"]   = Vector{UInt64}(0:M-1)
        g["Masses"]        = masses
        g["ReplicaIndex"]  = Vector{Int16}(crossings.replica_idx)
        g["ShellIndex"]    = Vector{Int16}(crossings.shell_idx)

        if v_mode == :radial
            g["RadialVelocity"] = v_radial
        else
            g["Velocities"] = v_full
        end

        if keep_particle_idx
            g["LagrangianParticleIndex"] = Vector{Int32}(crossings.particle_idx)
        end
    end
    return (n_particles=M, n_replicas=n_rep)
end
