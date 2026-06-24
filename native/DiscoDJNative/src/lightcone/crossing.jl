"""
Lightcone-crossing detection via Newton iteration.

For each particle (Lagrangian index) in each replica, find the scale factor
a_cross such that |x(a_cross) - observer| = χ(a_cross), where x(a) is the
LPT trajectory and χ(a) is the comoving distance.

Algorithm (matching DISCO-DJ):
1. Seed a_cross from the shell bin (log-linear interpolation)
2. Run n_newton_iters Newton–Raphson steps
3. Accept if |residual| < radial_residual_tol and a_cross ∈ [a_far, a_near]
"""

export find_lightcone_crossings

using LinearAlgebra
using StaticArrays

# ── Trajectory evaluation at scale factor a ───────────────────────────────────

function _trajectory(q::AbstractVector, psi1_flat::AbstractMatrix,
                     psi2_flat::Union{AbstractMatrix,Nothing},
                     cosmo::Cosmology, a::Real, pid::Int)
    D1 = growth_D1(cosmo, a)
    x  = q .+ D1 .* psi1_flat[pid, :]
    if psi2_flat !== nothing
        D2 = D1^2   # legacy ψ₂ carries the EdS -3/7 coefficient (see evaluate.jl)
        x .+= D2 .* psi2_flat[pid, :]
    end
    return x
end

function _trajectory_dot(psi1_flat::AbstractMatrix,
                         psi2_flat::Union{AbstractMatrix,Nothing},
                         cosmo::Cosmology, a::Real, pid::Int)
    # dx/da for Newton step
    H0 = 100 * cosmo.h
    D1 = growth_D1(cosmo, a)
    # dD1/da ≈ finite difference
    eps = 1e-5
    dD1 = (growth_D1(cosmo, a*(1+eps)) - growth_D1(cosmo, a*(1-eps))) / (2*a*eps)
    v  = dD1 .* psi1_flat[pid, :]
    if psi2_flat !== nothing
        dD2 = ((growth_D1(cosmo, a*(1+eps)))^2 -
               (growth_D1(cosmo, a*(1-eps)))^2) / (2*a*eps)   # d(D₁²)/da
        v .+= dD2 .* psi2_flat[pid, :]
    end
    return v
end

# ── dchi/da for Newton step on the χ(a) side ──────────────────────────────────
# χ(a) is the comoving distance to scale factor a, which DECREASES with a, so
# dχ/da = −(c/H₀)/(a²E) < 0 (matches JAX `dchi_da = -c_over_H0 / (a**2 * E_a)`).
# (The previous +(c/H₀)/(a²E) was d(χ_fwd)/da — the wrong sign — which inverted the
# χ-term of dF/da and made the Newton step move away from the crossing.)

function _dchi_da_at(cosmo::Cosmology, a::Real)
    return -2997.92458 / (a^2 * hubble_E(cosmo, a))
end

# ── Newton iteration for one particle × replica ───────────────────────────────

function _newton_crossing(q::AbstractVector, replica_offset::AbstractVector,
                          psi1_flat::AbstractMatrix,
                          psi2_flat::Union{AbstractMatrix,Nothing},
                          cosmo::Cosmology, a0::Real, observer::AbstractVector,
                          pid::Int; n_iters::Int=1)
    a = a0
    for _ in 1:n_iters
        x    = _trajectory(q, psi1_flat, psi2_flat, cosmo, a, pid) .+ replica_offset
        chi_a = comoving_distance(cosmo, a)
        d    = norm(x .- observer)

        # Residual: |x(a) - obs| - χ(a) = 0
        F    = d - chi_a

        # dF/da = (x-obs)·(dx/da) / d  - dχ/da
        xdot = _trajectory_dot(psi1_flat, psi2_flat, cosmo, a, pid)
        dFda = dot(x .- observer, xdot) / max(d, 1e-10) - _dchi_da_at(cosmo, a)

        abs(dFda) < 1e-15 && break
        a_new = a - F / dFda
        a = a_new
    end
    return a
end

# ── Robust crossing seed: bracket the root of F(a)=|x(a)-obs|-χ(a), then bisect ─
# Mirrors JAX's secant/bracket seed (shells act as brackets): scan F over
# [a_far, a_near], take the first sign change, and bisect to a tight seed — robust
# regardless of `n_newton_iters` (Newton then only polishes).  Returns NaN when the
# trajectory never crosses the shell in [a_far, a_near].
function _bracket_seed(q::AbstractVector, rep_offset::AbstractVector,
                       psi1_flat::AbstractMatrix, psi2_flat::Union{AbstractMatrix,Nothing},
                       cosmo::Cosmology, a_far::Real, a_near::Real,
                       observer::AbstractVector, pid::Int; n_scan::Int=16, n_bisect::Int=40)
    Fa(a) = norm(_trajectory(q, psi1_flat, psi2_flat, cosmo, a, pid) .+ rep_offset .- observer) -
            comoving_distance(cosmo, a)
    a_prev = a_far; F_prev = Fa(a_far)
    F_prev == 0 && return oftype(a_far, a_far)
    for s in 1:n_scan
        a_cur = a_far + (a_near - a_far) * s / n_scan
        F_cur = Fa(a_cur)
        if (F_prev < 0) != (F_cur < 0)              # sign change ⇒ bracket [a_prev, a_cur]
            lo, hi, Flo = a_prev, a_cur, F_prev
            for _ in 1:n_bisect
                mid  = (lo + hi) / 2
                Fmid = Fa(mid)
                Fmid == 0 && return mid
                if (Fmid < 0) == (Flo < 0)
                    lo = mid; Flo = Fmid
                else
                    hi = mid
                end
            end
            return (lo + hi) / 2
        end
        a_prev = a_cur; F_prev = F_cur
    end
    return oftype(a_far, NaN)                         # no crossing
end

# ── Main function: find all crossings for all particles × replicas ────────────

"""
    find_lightcone_crossings(lpt, cosmo, a_edges, observer, replica_offsets;
                             n_newton_iters=1, radial_residual_tol=0.1,
                             n_order=nothing) -> NamedTuple

Find lightcone crossings for all particles in all replicas.

Returns named tuple with:
- `particle_idx` — Lagrangian particle index (0-based, range [0, N³))
- `replica_idx`  — row index into `replica_offsets`
- `a_cross`      — crossing scale factor
- `x`            — position at crossing [Mpc/h]
- `shell_idx`    — shell bin index
"""
function find_lightcone_crossings(lpt::LPTResult{T}, cosmo::Cosmology{CT},
                                  a_edges::AbstractVector,
                                  observer::AbstractVector,
                                  replica_offsets::AbstractMatrix{Int};
                                  n_newton_iters::Int=1,
                                  radial_residual_tol::Real=0.1,
                                  n_order::Union{Int,Nothing}=nothing,
                                  boxsize::Real=lpt.boxsize) where {T, CT}
    res = lpt.res
    N   = res^3
    L   = boxsize

    # Flatten displacement fields to (N, 3)
    psi1_flat = reshape(lpt.psi1, N, 3)
    psi2_flat = lpt.psi2 !== nothing ? reshape(lpt.psi2, N, 3) : nothing

    # Lagrangian grid positions
    q_grid = lagrangian_grid(res, L; T)

    a_far  = T(a_edges[1])
    a_near = T(a_edges[end])
    n_rep  = size(replica_offsets, 1)

    # Pre-allocate results (upper bound: N × n_rep crossings)
    particle_idx_out = Vector{Int32}()
    replica_idx_out  = Vector{Int16}()
    a_cross_out      = Vector{T}()
    x_out            = Vector{SVector{3,T}}()
    shell_idx_out    = Vector{Int16}()

    for rep_i in 1:n_rep
        rep_offset = T.(replica_offsets[rep_i, :]) .* L

        for pid in 1:N
            q = SVector{3,T}(q_grid[pid, 1], q_grid[pid, 2], q_grid[pid, 3])

            # Robust seed: bracket the root of F(a)=|x(a)-obs|-χ(a) and bisect.
            # NaN ⇒ trajectory never crosses the shell (the correct rejection).
            a_seed = _bracket_seed(q, rep_offset, psi1_flat, psi2_flat,
                                   cosmo, a_far, a_near, observer, pid)
            isnan(a_seed) && continue

            # Optional Newton polish (the bracketed seed is already tight)
            a_cross = n_newton_iters > 0 ?
                _newton_crossing(q, rep_offset, psi1_flat, psi2_flat,
                                 cosmo, a_seed, observer, pid; n_iters=n_newton_iters) :
                a_seed

            # Validity checks
            a_cross < a_far  && continue
            a_cross > a_near && continue

            x_cross = _trajectory(q, psi1_flat, psi2_flat, cosmo, a_cross, pid) .+ rep_offset
            chi_cross = comoving_distance(cosmo, a_cross)
            residual  = abs(norm(x_cross .- observer) - chi_cross)
            residual > radial_residual_tol && continue

            sh_idx = shell_index_for_a(a_cross, a_edges)
            sh_idx < 0 && continue

            push!(particle_idx_out, Int32(pid - 1))   # 0-based
            push!(replica_idx_out, Int16(rep_i - 1))  # 0-based
            push!(a_cross_out, T(a_cross))
            push!(x_out, SVector{3,T}(x_cross...))
            push!(shell_idx_out, Int16(sh_idx - 1))   # 0-based
        end
    end

    # Build x matrix
    n_rows = length(particle_idx_out)
    x_mat  = Matrix{T}(undef, n_rows, 3)
    for i in 1:n_rows
        x_mat[i, 1] = x_out[i][1]
        x_mat[i, 2] = x_out[i][2]
        x_mat[i, 3] = x_out[i][3]
    end

    return (particle_idx = particle_idx_out,
            replica_idx  = replica_idx_out,
            a_cross      = a_cross_out,
            x            = x_mat,
            shell_idx    = shell_idx_out)
end
