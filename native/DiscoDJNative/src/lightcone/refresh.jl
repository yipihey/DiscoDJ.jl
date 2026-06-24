"""
Lightcone cosmology refresh — fast parameter sweeps without re-running LPT.

Implements both DISCO-DJ refresh modes:
- `mode=:fixed_psi` — reuse Ψ₁,Ψ₂ verbatim, only re-evaluate D_n(a) and χ(a)
  (~2.7× faster than from-scratch; approximate for T(k)-changing parameters)
- `mode=:exact`     — regenerate white noise from saved seed at new cosmology,
  recompute Ψ₁,Ψ₂, then Newton-refine (same cost as fresh run)

Both modes produce output identical in schema to `evaluate_lpt_lightcone_to_hdf5`.

Reference: DISCO-DJ lightcone integration guide §6.
"""

export load_lpt_scene, refresh_lightcone_cosmology, refresh_lightcone_arrays

using HDF5

# ── load_lpt_scene ────────────────────────────────────────────────────────────

"""
    load_lpt_scene(path; load_psi=true) -> Dict

Load the LPT scene file saved by `save_lpt_scene`.
Returns Dict with keys: "res", "boxsize", "n_order", "cosmology_params",
"transfer", "seed", "psi_1", "psi_2" (if saved).
"""
function load_lpt_scene(path::String; load_psi::Bool=true)
    scene = Dict{String,Any}()
    h5open(path, "r") do f
        g = f["Header"]
        a = attributes(g)
        scene["res"]     = read(a["res"])
        scene["boxsize"] = read(a["boxsize"])
        scene["n_order"] = read(a["n_order"])
        scene["transfer"] = read(a["transfer"])
        scene["seed"]     = read(a["seed"])
        scene["cosmology_params"] = Dict(
            "Omega_c" => read(a["cosmo_Omega_c"]),
            "Omega_b" => read(a["cosmo_Omega_b"]),
            "h"       => read(a["cosmo_h"]),
            "sigma8"  => read(a["cosmo_sigma8"]),
            "n_s"     => read(a["cosmo_n_s"]),
            "Omega_k" => read(a["cosmo_Omega_k"]),
            "w0"      => read(a["cosmo_w0"]),
            "wa"      => read(a["cosmo_wa"]),
        )
        if load_psi
            haskey(f, "psi_1") && (scene["psi_1"] = read(f["psi_1"]))
            haskey(f, "psi_2") && (scene["psi_2"] = read(f["psi_2"]))
            haskey(f, "psi_3") && (scene["psi_3"] = read(f["psi_3"]))
        end
    end
    return scene
end

# ── Newton iteration ──────────────────────────────────────────────────────────

# Single-particle Newton iteration for lightcone crossing at new cosmology.
# `psi1_p` and `psi2_p` are pre-sliced 3-vectors for particle pid.
function _newton_1d(q, rep_off, psi1_p, psi2_p, cosmo, a0, observer; n_iters)
    T    = Float64
    a    = T(a0)
    obs  = T.(observer)
    qr   = T.(q) .+ T.(rep_off)
    p1   = T.(psi1_p)
    p2   = psi2_p !== nothing ? T.(psi2_p) : nothing

    for _ in 1:n_iters
        D1    = T(growth_D1(cosmo, a))
        x     = qr .+ D1 .* p1
        p2 !== nothing && (x .+= T(growth_D2(cosmo, a)) * D1^2 .* p2)
        dist  = norm(x .- obs)
        chi   = T(comoving_distance(cosmo, a))
        resid = dist - chi

        # Finite-difference derivative dr/da
        da  = a * T(1e-4)
        a2  = min(a + da, T(1.0))
        D1p = T(growth_D1(cosmo, a2))
        xp  = qr .+ D1p .* p1
        p2 !== nothing && (xp .+= T(growth_D2(cosmo, a2)) * D1p^2 .* p2)
        dr_da = ((norm(xp .- obs) - T(comoving_distance(cosmo, a2))) - resid) / (a2 - a)

        abs(dr_da) < T(1e-30) && break
        a = clamp(a - resid / dr_da, T(0.01), T(1.0))
    end
    return a
end

function _gadget_velocity_flat(psi1, psi2, cosmo, pid, a)
    H0 = 100 * cosmo.h
    E  = hubble_E(cosmo, a)
    f1 = growth_f1(cosmo, a)
    D1 = growth_D1(cosmo, a)
    v  = f1 * D1 * E * H0 .* psi1[pid, :]
    if psi2 !== nothing
        D2 = growth_D2(cosmo, a) * D1^2
        v .+= 2f1 * D2 * E * H0 .* psi2[pid, :]
    end
    return v .* (100 / a^1.5)
end

# ── refresh_lightcone_arrays (in-memory, autodiff-compatible) ─────────────────

"""
    refresh_lightcone_arrays(particle_idx, replica_idx, a_cross_seed,
                             q_flat, psi_tuple, replica_offsets, observer,
                             a_shells, cosmo; a_far, a_near, n_newton_iters=1,
                             v_mode=:radial) -> NamedTuple

Pure in-memory cosmology refresh: given saved (Ψ₁, Ψ₂) and seeded crossings
from a fiducial run, Newton-refine each crossing at the new cosmology.

`psi_tuple = (psi_1, psi_2)` — (N,3) flat displacement arrays.
Returns: (x, v_radial, a_cross, shell_idx, valid).
"""
function refresh_lightcone_arrays(particle_idx::AbstractVector{Int32},
                                   replica_idx::AbstractVector{Int16},
                                   a_cross_seed::AbstractVector,
                                   q_flat::AbstractMatrix,
                                   psi_tuple::Tuple,
                                   replica_offsets::AbstractMatrix,
                                   observer::AbstractVector,
                                   a_shells::AbstractVector,
                                   cosmo::Cosmology;
                                   a_far::Real, a_near::Real,
                                   n_newton_iters::Int=1,
                                   v_mode::Symbol=:radial)
    M = length(particle_idx)
    T = eltype(a_cross_seed)
    psi1_flat = psi_tuple[1]
    psi2_flat = length(psi_tuple) >= 2 ? psi_tuple[2] : nothing

    x_out        = Matrix{T}(undef, M, 3)
    a_cross_out  = Vector{T}(undef, M)
    v_radial_out = Vector{T}(undef, M)
    shell_out    = Vector{Int16}(undef, M)
    valid_out    = Vector{Bool}(undef, M)   # Bool not BitVector: bit-packing races with @threads

    Threads.@threads for i in 1:M
        pid   = Int(particle_idx[i]) + 1   # 1-based
        rep_i = Int(replica_idx[i]) + 1
        a0    = T(a_cross_seed[i])

        rep_off = T.(replica_offsets[rep_i, :]) .* T(size(q_flat, 1)^(1/3))
        q       = T.(q_flat[pid, :])

        # Pre-slice displacements for this particle
        psi1_p = psi1_flat[pid, :]
        psi2_p = psi2_flat !== nothing ? psi2_flat[pid, :] : nothing

        a_cross = T(_newton_1d(q, rep_off, psi1_p, psi2_p, cosmo, a0, observer;
                                n_iters=n_newton_iters))

        valid = a_far <= a_cross <= a_near
        valid_out[i] = valid

        if valid
            D1 = growth_D1(cosmo, a_cross)
            x  = q .+ T(D1) .* psi1_p
            if psi2_flat !== nothing
                D2 = growth_D2(cosmo, a_cross) * D1^2
                x .+= T(D2) .* psi2_p
            end
            x .+= rep_off
        else
            x = fill(T(NaN), 3)
        end

        x_out[i, :] = x
        a_cross_out[i] = a_cross
        shell_out[i] = Int16(shell_index_for_a(a_cross, a_shells) - 1)

        # Radial velocity
        if valid
            n̂ = (x .- T.(observer)) ./ max(norm(x .- T.(observer)), T(1e-10))
            v = _gadget_velocity_flat(psi1_flat, psi2_flat, cosmo, pid, a_cross)
            v_radial_out[i] = dot(T.(v), n̂)
        else
            v_radial_out[i] = T(NaN)
        end
    end

    return (x=x_out, v_radial=v_radial_out, a_cross=a_cross_out,
            shell_idx=shell_out, valid=valid_out)
end

# ── refresh_lightcone_cosmology (file → file) ─────────────────────────────────

"""
    refresh_lightcone_cosmology(; scene_path, input_lightcone, output_lightcone,
                                 new_cosmology, mode=:fixed_psi,
                                 sigma8_rescale=true, n_newton_iters=1,
                                 compression=:zstd, verbose=false) -> NamedTuple

Refresh a lightcone catalogue at a new cosmology.

`mode=:fixed_psi` — fastest: reuse stored Ψ and re-evaluate growth+χ.
`mode=:exact`     — regenerate Ψ from seed at new cosmology.

Returns (n_particles_in, n_particles_out, n_replicas, mode).
"""
function refresh_lightcone_cosmology(; scene_path::String, input_lightcone::String,
                                       output_lightcone::String,
                                       new_cosmology::Cosmology,
                                       mode::Symbol=:fixed_psi,
                                       sigma8_rescale::Bool=true,
                                       n_newton_iters::Int=1,
                                       compression::Symbol=:zstd,
                                       verbose::Bool=false)
    scene = load_lpt_scene(scene_path; load_psi=true)
    meta  = read_lightcone_header(input_lightcone)

    n_in  = meta["n_particles"]
    verbose && println("Refreshing $n_in particles in mode=$mode")

    # Read input catalogue
    pid = Int32[]; rep = Int16[]; a_seed = Float32[]
    h5open(input_lightcone, "r") do f
        g = f["PartType1"]
        pid    = read(g["LagrangianParticleIndex"])
        rep    = read(g["ReplicaIndex"])
        a_seed = read(g["ScaleFactor"])
    end

    res     = scene["res"]
    boxsize = scene["boxsize"]
    n_order = scene["n_order"]

    if mode == :fixed_psi
        psi1 = reshape(Float32.(scene["psi_1"]), res^3, 3)
        psi2 = haskey(scene, "psi_2") ? reshape(Float32.(scene["psi_2"]), res^3, 3) : nothing

        if sigma8_rescale
            sigma8_fid = Float32(scene["cosmology_params"]["sigma8"])
            sigma8_new = Float32(new_cosmology.sigma8)
            psi1 .*= (sigma8_new / sigma8_fid)
            psi2 !== nothing && (psi2 .*= (sigma8_new / sigma8_fid))
        end

        psi_tuple = psi2 !== nothing ? (psi1, psi2) : (psi1,)

    elseif mode == :exact
        T_fp = Float32
        pk_new = linear_power_spectrum(new_cosmology; transfer=scene["transfer"])
        fphi   = generate_grf(:ngenic, 3, pk_new, res, Float64(boxsize), scene["seed"];
                               dtype=T_fp, dtype_c=Complex{T_fp})
        grid   = get_fourier_grid(res, boxsize; T=T_fp)
        lpt_new = compute_lpt(fphi, grid; n_order=n_order, backend=:threads)
        psi1 = reshape(lpt_new.psi1, res^3, 3)
        psi2 = lpt_new.psi2 !== nothing ? reshape(lpt_new.psi2, res^3, 3) : nothing
        psi_tuple = psi2 !== nothing ? (psi1, psi2) : (psi1,)
    else
        error("mode must be :fixed_psi or :exact")
    end

    q_flat = lagrangian_grid(res, boxsize; T=Float64)

    rep_offs = zeros(Int, max(maximum(Int.(rep))+1, 1), 3)

    a_shells = Float32.(collect(LinRange(minimum(a_seed), maximum(a_seed), 65)))

    observer_pos = get(meta, "Observer", Float64[boxsize/2, boxsize/2, boxsize/2])

    result = refresh_lightcone_arrays(pid, rep, a_seed, q_flat, psi_tuple,
                                       rep_offs, observer_pos, a_shells, new_cosmology;
                                       a_far=Float64(minimum(a_seed)),
                                       a_near=Float64(maximum(a_seed)),
                                       n_newton_iters=n_newton_iters)

    valid_mask = result.valid
    n_out = sum(valid_mask)
    verbose && println("Writing $n_out particles ($(round(100*n_out/n_in, digits=1))% of input)")

    # Write refreshed catalogue to HDF5
    n_per_rep = res^3
    n_rep     = max(maximum(Int.(rep[valid_mask]); init=0) + 1, 1)
    rho_crit  = 2.775e11   # h² M☉/Mpc³
    mass_unit = Omega_m(new_cosmology) * rho_crit * Float64(boxsize)^3 / n_per_rep / 1e10

    h5open(output_lightcone, "w") do f
        _write_header!(f, new_cosmology, observer_pos, Float64(boxsize), n_out,
                       n_per_rep, n_rep, true)
        g = create_group(f, "PartType1")
        g["Coordinates"]             = Matrix{Float32}(result.x[valid_mask, :])
        g["RadialVelocity"]          = Vector{Float32}(result.v_radial[valid_mask])
        g["ScaleFactor"]             = Vector{Float32}(result.a_cross[valid_mask])
        g["ParticleIDs"]             = Vector{UInt64}(0:n_out-1)
        g["Masses"]                  = fill(Float32(mass_unit), n_out)
        g["ReplicaIndex"]            = Vector{Int16}(rep[valid_mask])
        g["ShellIndex"]              = Vector{Int16}(result.shell_idx[valid_mask])
        g["LagrangianParticleIndex"] = Vector{Int32}(pid[valid_mask])
    end

    return (n_particles_in=n_in, n_particles_out=n_out, n_replicas=n_rep, mode=mode)
end
