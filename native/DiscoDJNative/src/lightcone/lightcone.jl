"""
High-level lightcone evaluation API.

Matches the Python DISCO-DJ surface:
- `evaluate_lpt_lightcone`         — in-memory streaming result
- `evaluate_lpt_lightcone_to_hdf5` — streaming + HDF5 output

The DiscoDJScene struct bundles the built LPT + cosmology like Python's
`DiscoDJ` object after `.with_lpt()`.
"""

export DiscoDJScene, build_scene, evaluate_lpt_lightcone, evaluate_lpt_lightcone_to_hdf5,
       save_lpt_scene

using LinearAlgebra

# ── DiscoDJScene ──────────────────────────────────────────────────────────────

"""
    DiscoDJScene

Equivalent of Python's built `DiscoDJ` object after `with_lpt()`.
Holds everything needed to evaluate displacements and lightcones.
"""
struct DiscoDJScene{T<:AbstractFloat}
    lpt::LPTResult{T}
    cosmo::Cosmology{T}
    pk_table::Dict{String,Vector{Float64}}
    spec::NamedTuple    # dim, res, boxsize, n_order, seed, transfer, precision
end

"""
    build_scene(; dim=3, res=64, boxsize=100.0, cosmo="Planck18EEBAOSN",
                  n_order=2, seed=42, transfer="Eisenstein-Hu",
                  precision=:single, backend=:ka) -> DiscoDJScene

Construct the full DISCO-DJ pipeline (timetables → P(k) → ICs → nLPT).
Matches `DiscoDJLib.build(spec)` from the Python binding.
"""
function build_scene(; dim::Int=3, res::Int=64, boxsize::Float64=100.0,
                       cosmo::Union{String,Cosmology}="Planck18EEBAOSN",
                       n_order::Int=2, seed::Int=42,
                       transfer::String="Eisenstein-Hu",
                       precision::Symbol=:single,
                       backend::Symbol=:ka)
    T = precision == :single ? Float32 : Float64

    # 1. Cosmology + timetables
    c = cosmo isa String ? Cosmology(cosmo; T=Float64) : cosmo

    # 2. Linear P(k)
    pk_table = linear_power_spectrum(c; transfer=transfer)

    # 3. ICs
    fphi_ini = generate_grf(:ngenic, dim, pk_table, res, boxsize, seed;
                             dtype=T, dtype_c=Complex{T})

    # 4. Fourier grid
    grid = get_fourier_grid(res, boxsize; T=T)

    # 5. nLPT
    lpt = compute_lpt(fphi_ini, grid; n_order=n_order, backend=backend)

    spec = (dim=dim, res=res, boxsize=boxsize, n_order=n_order,
            seed=seed, transfer=transfer, precision=precision)
    return DiscoDJScene{T}(lpt, c, pk_table, spec)
end

# ── evaluate_lpt_lightcone ────────────────────────────────────────────────────

"""
    evaluate_lpt_lightcone(scene; a_far, a_near=1.0, n_shells=64,
                           observer=nothing, n_newton_iters=1,
                           radial_residual_tol=0.1,
                           keep_particle_idx=true, v_mode=:radial,
                           streaming=true) -> NamedTuple

Evaluate the LPT past-lightcone between `a_far` and `a_near`.

Returns named tuple with:
- `x`            — (M,3) positions [Mpc/h]
- `v_radial`     — (M,) Gadget √a-scaled radial velocity (if v_mode=:radial)
- `velocities`   — (M,3) full velocity (if v_mode=:full)
- `a_cross`      — (M,) crossing scale factors
- `shell_idx`    — (M,) shell indices
- `replica_idx`  — (M,) replica indices
- `particle_idx` — (M,) Lagrangian indices (if keep_particle_idx)
"""
function evaluate_lpt_lightcone(scene::DiscoDJScene{T};
                                  a_far::Real, a_near::Real=1.0, n_shells::Int=64,
                                  observer::Union{AbstractVector,Nothing}=nothing,
                                  n_newton_iters::Int=1,
                                  radial_residual_tol::Real=0.1,
                                  keep_particle_idx::Bool=true,
                                  v_mode::Symbol=:radial,
                                  kwargs...) where T
    L   = T(scene.lpt.boxsize)
    obs = observer === nothing ? T[L/2, L/2, L/2] : T.(observer)

    a_edges   = log_shells(T(a_far), T(a_near), n_shells)
    chi_near  = comoving_distance(scene.cosmo, a_near)
    chi_far   = comoving_distance(scene.cosmo, a_far)
    rep_offs  = enumerate_replicas(L, obs, chi_near, chi_far)

    crossings = find_lightcone_crossings(
        scene.lpt, scene.cosmo, a_edges, obs, rep_offs;
        n_newton_iters=n_newton_iters,
        radial_residual_tol=radial_residual_tol,
    )

    # Compute velocities
    M         = length(crossings.a_cross)
    n_per_rep = scene.lpt.res^3
    psi1_flat = reshape(scene.lpt.psi1, n_per_rep, 3)
    psi2_flat = scene.lpt.psi2 !== nothing ? reshape(scene.lpt.psi2, n_per_rep, 3) : nothing

    v_radial_vec = Vector{T}(undef, M)
    v_full_mat   = v_mode == :full ? Matrix{T}(undef, M, 3) : nothing

    for i in 1:M
        pid = Int(crossings.particle_idx[i]) + 1
        a   = crossings.a_cross[i]
        v   = _gadget_velocity(scene.lpt, scene.cosmo, psi1_flat, psi2_flat, pid, a)
        x   = crossings.x[i, :]
        n̂   = (x .- obs) ./ max(norm(x .- obs), T(1e-10))
        v_radial_vec[i] = dot(v, n̂)
        v_mode == :full && (v_full_mat[i, :] = v)
    end

    result = (
        x            = crossings.x,
        v_radial     = v_radial_vec,
        a_cross      = crossings.a_cross,
        shell_idx    = crossings.shell_idx,
        replica_idx  = crossings.replica_idx,
    )
    if keep_particle_idx
        result = merge(result, (particle_idx = crossings.particle_idx,))
    end
    if v_mode == :full
        result = merge(result, (velocities = v_full_mat,))
    end
    return result
end

function _gadget_velocity(lpt, cosmo, psi1_flat, psi2_flat, pid, a)
    H0 = 100 * cosmo.h
    E  = hubble_E(cosmo, a)
    f1 = growth_f1(cosmo, a)
    D1 = growth_D1(cosmo, a)
    H  = E * H0
    v  = f1 * D1 * H .* psi1_flat[pid, :]
    if psi2_flat !== nothing
        D2 = growth_D2(cosmo, a) * D1^2
        v .+= 2f1 * D2 * H .* psi2_flat[pid, :]
    end
    return v .* (100 / a^1.5)   # Gadget convention
end

"""
    evaluate_lpt_lightcone_to_hdf5(scene, path; kwargs...) -> NamedTuple

Write the lightcone catalogue directly to HDF5.
All kwargs forwarded to `evaluate_lpt_lightcone` and `write_lightcone_hdf5`.
"""
function evaluate_lpt_lightcone_to_hdf5(scene::DiscoDJScene{T}, path::String;
                                          observer::Union{AbstractVector,Nothing}=nothing,
                                          a_far::Real, a_near::Real=1.0, n_shells::Int=64,
                                          keep_particle_idx::Bool=true,
                                          v_mode::Symbol=:radial,
                                          compression::Symbol=:zstd,
                                          kwargs...) where T
    obs = observer === nothing ? T[scene.lpt.boxsize/2, scene.lpt.boxsize/2, scene.lpt.boxsize/2] : T.(observer)
    lc  = evaluate_lpt_lightcone(scene; a_far=a_far, a_near=a_near, n_shells=n_shells,
                                   observer=obs, keep_particle_idx=keep_particle_idx,
                                   v_mode=v_mode, kwargs...)
    lc_nt = (x=lc.x, a_cross=lc.a_cross,
             particle_idx=keep_particle_idx ? lc.particle_idx : nothing,
             replica_idx=lc.replica_idx, shell_idx=lc.shell_idx)

    summary = write_lightcone_hdf5(path, lc_nt, scene.lpt, scene.cosmo, obs;
                                    v_mode=v_mode, keep_particle_idx=keep_particle_idx,
                                    compression=compression)
    return merge(summary, (lc_mode="lpt",))
end

# ── save_lpt_scene ────────────────────────────────────────────────────────────

"""
    save_lpt_scene(scene, path; include_psi=true) -> nothing

Persist the LPT displacement fields and metadata to HDF5 for later
cosmology refresh (matched to DISCO-DJ's save_lpt_scene schema).
"""
function save_lpt_scene(scene::DiscoDJScene{T}, path::String;
                         include_psi::Bool=true) where T
    h5open(path, "w") do f
        c = scene.cosmo
        g = create_group(f, "Header")
        a = attributes(g)
        a["res"]      = scene.lpt.res
        a["boxsize"]  = scene.lpt.boxsize
        a["n_order"]  = scene.lpt.n_order
        a["cosmo_Omega_c"] = c.Omega_c; a["cosmo_Omega_b"] = c.Omega_b
        a["cosmo_h"]       = c.h;       a["cosmo_sigma8"]  = c.sigma8
        a["cosmo_n_s"]     = c.n_s;     a["cosmo_Omega_k"] = c.Omega_k
        a["cosmo_w0"]      = c.w0;      a["cosmo_wa"]      = c.wa
        a["transfer"]      = scene.spec.transfer
        a["seed"]          = scene.spec.seed

        if include_psi
            f["psi_1"] = scene.lpt.psi1
            scene.lpt.psi2 !== nothing && (f["psi_2"] = scene.lpt.psi2)
            scene.lpt.psi3 !== nothing && (f["psi_3"] = scene.lpt.psi3)
        end
    end
    return nothing
end
