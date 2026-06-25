"""
Differentiable mass deposit onto an Eulerian mesh.

Two estimators, both differentiable w.r.t. particle positions and weights:

  * `cic_deposit(pos, w, res, boxsize)` — Cloud-In-Cell particle-mesh deposit.
    The single hand-written adjoint (`ChainRulesCore.rrule`): the VJP gathers the
    mesh cotangent with the CIC weights (→ `w̄`) and with the CIC *weight gradients*
    (→ `x̄`), exactly the JAX `scatter_and_gather` VJP.

  * `sheet_deposit(x_grid, w_grid, res, boxsize; n_sub)` — the tetrahedral cold-dark-
    matter phase-space-sheet density (Abel, Hahn & Kaehler 2012).  Each Lagrangian
    cube of the displaced grid is split into 6 tetrahedra; each tetrahedron's mass is
    spread over its Eulerian volume by depositing `n_sub` barycentric sample points
    through `cic_deposit`.  This is a *differentiable composition* on top of CIC (the
    sample points are linear in the cube-corner positions), so the only new adjoint is
    the CIC one above.  Pre-shell-crossing this gives the smooth single-stream density;
    where the sheet folds it correctly multi-counts streams.

`x_grid` / `w_grid` are `(res,res,res,·)` arrays in the same layout as the nLPT
displacement (the irfft grid); `lagrangian_grid_3d` returns the matching `q`.
"""

export cic_deposit, sheet_deposit, lagrangian_grid_3d

using ChainRulesCore
import ChainRulesCore: rrule, NoTangent, unthunk

# ── Lagrangian grid in (res,res,res,3) layout (matches the nLPT ψ array) ──────
function lagrangian_grid_3d(res::Int, boxsize::Real; T::Type{<:AbstractFloat}=Float64)
    dx = T(boxsize) / res
    q = Array{T,4}(undef, res, res, res, 3)
    @inbounds for k in 1:res, j in 1:res, i in 1:res
        q[i, j, k, 1] = (i - 1) * dx
        q[i, j, k, 2] = (j - 1) * dx
        q[i, j, k, 3] = (k - 1) * dx
    end
    return q
end

# Per-axis CIC stencil: 1-based neighbor cells (periodic) + fraction f.
@inline function _cic_ax(x, invdx, res::Int)
    u  = x * invdx
    fl = floor(u)
    f  = u - fl
    i0 = mod(Int(fl), res)
    return (i0 + 1, mod(i0 + 1, res) + 1, f)
end

# ── CIC deposit (KernelAbstractions: one code path on CPU and CUDA) ───────────
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize

# Forward scatter: atomic-add each particle's 8 CIC contributions into the mesh.
@kernel function _cic_scatter_kernel!(mesh, @Const(pos), @Const(w), res::Int, invdx)
    p = @index(Global)
    @inbounds if p <= size(pos, 1)
        a0x, a1x, fx = _cic_ax(pos[p, 1], invdx, res)
        a0y, a1y, fy = _cic_ax(pos[p, 2], invdx, res)
        a0z, a1z, fz = _cic_ax(pos[p, 3], invdx, res)
        wp = w[p]; gx0 = 1 - fx; gx1 = fx; gy0 = 1 - fy; gy1 = fy; gz0 = 1 - fz; gz1 = fz
        KernelAbstractions.@atomic mesh[a0x, a0y, a0z] += wp * gx0 * gy0 * gz0
        KernelAbstractions.@atomic mesh[a1x, a0y, a0z] += wp * gx1 * gy0 * gz0
        KernelAbstractions.@atomic mesh[a0x, a1y, a0z] += wp * gx0 * gy1 * gz0
        KernelAbstractions.@atomic mesh[a1x, a1y, a0z] += wp * gx1 * gy1 * gz0
        KernelAbstractions.@atomic mesh[a0x, a0y, a1z] += wp * gx0 * gy0 * gz1
        KernelAbstractions.@atomic mesh[a1x, a0y, a1z] += wp * gx1 * gy0 * gz1
        KernelAbstractions.@atomic mesh[a0x, a1y, a1z] += wp * gx0 * gy1 * gz1
        KernelAbstractions.@atomic mesh[a1x, a1y, a1z] += wp * gx1 * gy1 * gz1
    end
end

# Adjoint gather: each particle reads its 8 mesh-cotangents (no atomics) → x̄, w̄.
@kernel function _cic_gather_kernel!(x̄, w̄, @Const(Δm), @Const(pos), @Const(w), res::Int, invdx)
    p = @index(Global)
    @inbounds if p <= size(pos, 1)
        a0x, a1x, fx = _cic_ax(pos[p, 1], invdx, res)
        a0y, a1y, fy = _cic_ax(pos[p, 2], invdx, res)
        a0z, a1z, fz = _cic_ax(pos[p, 3], invdx, res)
        wp = w[p]; gx0 = 1 - fx; gx1 = fx; gy0 = 1 - fy; gy1 = fy; gz0 = 1 - fz; gz1 = fz
        c000 = Δm[a0x,a0y,a0z]; c100 = Δm[a1x,a0y,a0z]; c010 = Δm[a0x,a1y,a0z]; c110 = Δm[a1x,a1y,a0z]
        c001 = Δm[a0x,a0y,a1z]; c101 = Δm[a1x,a0y,a1z]; c011 = Δm[a0x,a1y,a1z]; c111 = Δm[a1x,a1y,a1z]
        w̄[p] = c000*gx0*gy0*gz0 + c100*gx1*gy0*gz0 + c010*gx0*gy1*gz0 + c110*gx1*gy1*gz0 +
               c001*gx0*gy0*gz1 + c101*gx1*gy0*gz1 + c011*gx0*gy1*gz1 + c111*gx1*gy1*gz1
        s1 = (c100*gy0*gz0 + c110*gy1*gz0 + c101*gy0*gz1 + c111*gy1*gz1 -
              c000*gy0*gz0 - c010*gy1*gz0 - c001*gy0*gz1 - c011*gy1*gz1) * invdx
        s2 = (c010*gx0*gz0 + c110*gx1*gz0 + c011*gx0*gz1 + c111*gx1*gz1 -
              c000*gx0*gz0 - c100*gx1*gz0 - c001*gx0*gz1 - c101*gx1*gz1) * invdx
        s3 = (c001*gx0*gy0 + c101*gx1*gy0 + c011*gx0*gy1 + c111*gx1*gy1 -
              c000*gx0*gy0 - c100*gx1*gy0 - c010*gx0*gy1 - c110*gx1*gy1) * invdx
        x̄[p,1] = wp * s1; x̄[p,2] = wp * s2; x̄[p,3] = wp * s3
    end
end

"""
    cic_deposit(pos::(N,3), w::(N,), res, boxsize) -> mesh::(res,res,res)

Periodic Cloud-In-Cell deposit of weights `w` at positions `pos` (Mpc/h, wrapped to
[0,L)).  Runs on CPU or CUDA (KernelAbstractions); differentiable w.r.t. `pos` and `w`.
"""
function cic_deposit(pos::AbstractMatrix{T}, w::AbstractVector{T},
                     res::Int, boxsize::Real) where {T}
    backend = get_backend(pos)
    mesh = KernelAbstractions.zeros(backend, T, res, res, res)
    invdx = T(res) / T(boxsize)
    _cic_scatter_kernel!(backend)(mesh, pos, w, res, invdx; ndrange=size(pos, 1))
    synchronize(backend)
    return mesh
end

function rrule(::typeof(cic_deposit), pos::AbstractMatrix{T}, w::AbstractVector{T},
               res::Int, boxsize::Real) where {T}
    mesh = cic_deposit(pos, w, res, boxsize)
    function cic_pullback(Δ)
        Δm = unthunk(Δ)
        backend = get_backend(pos)
        x̄ = KernelAbstractions.zeros(backend, T, size(pos, 1), 3)
        w̄ = KernelAbstractions.zeros(backend, T, length(w))
        invdx = T(res) / T(boxsize)
        _cic_gather_kernel!(backend)(x̄, w̄, Δm, pos, w, res, invdx; ndrange=size(pos, 1))
        synchronize(backend)
        return (NoTangent(), x̄, w̄, NoTangent(), NoTangent())
    end
    return mesh, cic_pullback
end

# ── Tetrahedral phase-space-sheet deposit ─────────────────────────────────────
# Cube corners labelled (a,b,c)∈{0,1}³; the 6 tetrahedra share the main diagonal
# 0=(0,0,0) — 7=(1,1,1) (Kuhn/standard decomposition).
const _CUBE_CORNERS = ((0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1))
const _TETS = ((1,2,4,8), (1,4,3,8), (1,3,7,8), (1,7,5,8), (1,5,6,8), (1,6,2,8))  # 1-based corner ids

# Circular-shift the displaced grid so element [i,j,k] holds corner (a,b,c) of the
# cube anchored at (i,j,k): corner value = x_grid[i+a, j+b, k+c] (periodic).
_corner(xg::AbstractArray{T,4}, a, b, c) where {T} =
    circshift(xg, (-a, -b, -c, 0))

"""
    sheet_deposit(x_grid, w_grid, res, boxsize; n_sub=1) -> mesh::(res,res,res)

Tetrahedral CDM-sheet mass deposit.  `x_grid::(res,res,res,3)` are the Eulerian
positions of the Lagrangian vertices (e.g. `q .+ ψ`); `w_grid::(res,res,res)` are
per-vertex masses/weights (e.g. the bias weight `w(q)`; pass `ones` for matter).
`n_sub` barycentric sample points per tetrahedron spread its mass over the Eulerian
volume (`n_sub=1` = centroid; larger = better volume sampling).  Differentiable
w.r.t. `x_grid` and `w_grid`.
"""
function sheet_deposit(x_grid::AbstractArray{T,4}, w_grid::AbstractArray{T,3},
                       res::Int, boxsize::Real; n_sub::Int=1) where {T}
    corners = ntuple(8) do n
        a, b, c = _CUBE_CORNERS[n]
        _corner(x_grid, a, b, c)                               # (res,res,res,3)
    end
    wcorners = ntuple(8) do n
        a, b, c = _CUBE_CORNERS[n]
        circshift(w_grid, (-a, -b, -c))                        # (res,res,res)
    end
    bary = _tet_barycentric(T, n_sub)                          # n_sub × 4 weights
    m_tet = T(1) / 6                                           # cube mass 1 → 6 tets
    npts  = res^3
    # Deposit each (tetrahedron, sub-sample) and sum the meshes (Zygote-traceable:
    # `map` over a constant index list + `sum`, no in-place accumulation).
    pairs = @ignore_derivatives [(tet, s) for tet in _TETS for s in 1:n_sub]
    function _tet_sample_mesh(ts)
        (v1, v2, v3, v4), s = ts
        wt = (wcorners[v1] .+ wcorners[v2] .+ wcorners[v3] .+ wcorners[v4]) ./ 4
        b1, b2, b3, b4 = bary[s, 1], bary[s, 2], bary[s, 3], bary[s, 4]
        sp = b1 .* corners[v1] .+ b2 .* corners[v2] .+
             b3 .* corners[v3] .+ b4 .* corners[v4]            # (res,res,res,3)
        cic_deposit(reshape(sp, npts, 3), (m_tet / n_sub) .* reshape(wt, npts), res, boxsize)
    end
    return sum(map(_tet_sample_mesh, pairs))
end

# Barycentric coordinates of the n_sub sample points inside a tetrahedron.
# n_sub=1 → centroid; n_sub=4 → the 4 points halfway from centroid to each vertex
# (a symmetric volume sampling that still sums to the centroid).
function _tet_barycentric(::Type{T}, n_sub::Int) where {T}
    if n_sub == 1
        return fill(T(1) / 4, 1, 4)
    elseif n_sub == 4
        c = T(1) / 4
        b = Matrix{T}(undef, 4, 4)
        @inbounds for v in 1:4, j in 1:4
            b[v, j] = (j == v) ? (c + T(1)/2 * (1 - c)) : (c - T(1)/2 * c)
        end
        return b
    else
        error("n_sub must be 1 or 4")
    end
end
