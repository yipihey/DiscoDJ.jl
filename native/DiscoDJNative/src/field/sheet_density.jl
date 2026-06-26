"""
Grid-free AHK sheet density — the per-tet differentiable core (P1).

The phase-space sheet is the Lagrangian grid tessellated into 6 tetrahedra per cube
(`_TETS`/`_CUBE_CORNERS`).  Each tet's 4 vertices are the advected/lightcone-crossed
positions `x_grid` (layout `(res,res,res,3)`).  Connectivity is **non-periodic** — only
the (res−1)³ interior cubes are tessellated, since the lightcone embedding is not periodic
and boundary-wrapping tets would span the box (spurious volumes + lookup AABBs); the
dropped boundary layer is outside the survey footprint.  Per tet:

    V_T = (1/6)·det[y₂−y₁, y₃−y₁, y₄−y₁]     (signed Eulerian volume)
    w_T = (w₁+w₂+w₃+w₄)/4                      (mean bias weight)
    ρ_T = m_T·w_T / max(|V_T|, V_floor)        (m_T = 1/6; floor regularises caustics)

This file builds and FD-validates that per-tet machinery — the analytic volume det and
its **cofactor adjoint** (the new ingredient), the weight mean, and the |V_T| floor —
exposed through two scalar reductions:

    D = Σ_T ρ_T            (diagnostic: exercises the V_T → x_grid gradient)
    Z = Σ_T m_T·w_T        (the likelihood normalisation: exercises the w gradient)

No spatial lookup yet (P2) and no galaxy scatter yet (P3); the scatter will reuse exactly
this V_T/cofactor/w_T core, replacing the Σ_T reduction with a tet→galaxy deposit.  One
hand-written `rrule` (recompute-in-backward, à la `cic_deposit`): forward atomic-reduces
D,Z; backward recomputes each tet and scatters the det cofactors → `x̄_grid` and the
weight mean → `w̄`.  Runs on CPU and CUDA via KernelAbstractions.
"""

export sheet_tet_reduce, tet_volume_sum

# Combine _TETS (corner ids per tet) + _CUBE_CORNERS (id → (a,b,c) offset) into direct
# (6,4,3) integer offsets, so the kernel indexes an array (GPU-safe) not a runtime tuple.
function _build_tet_offsets()
    off = Array{Int}(undef, 6, 4, 3)
    @inbounds for t in 1:6, v in 1:4
        a, b, c = _CUBE_CORNERS[_TETS[t][v]]
        off[t, v, 1] = a; off[t, v, 2] = b; off[t, v, 3] = c
    end
    return off
end
const _TET_OFFSETS = _build_tet_offsets()

# Copy the tiny offset table to `ref`'s backend (constant connectivity, off the AD tape).
_offsets_on(ref::AbstractArray) = (y = similar(ref, Int, 6, 4, 3); copyto!(y, _TET_OFFSETS); y)

# ── forward: per-tet (V_T, w_T, ρ_T) reduced into the scalars D, Z (F64 accumulators) ──
@kernel function _sheet_reduce_fwd!(D, Z, @Const(xg), @Const(wg), @Const(off),
                                    res::Int, mT, floorvol)
    t, i, j, k = @index(Global, NTuple)
    @inbounds begin
        p1 = mod1(i+off[t,1,1],res); q1 = mod1(j+off[t,1,2],res); r1 = mod1(k+off[t,1,3],res)
        p2 = mod1(i+off[t,2,1],res); q2 = mod1(j+off[t,2,2],res); r2 = mod1(k+off[t,2,3],res)
        p3 = mod1(i+off[t,3,1],res); q3 = mod1(j+off[t,3,2],res); r3 = mod1(k+off[t,3,3],res)
        p4 = mod1(i+off[t,4,1],res); q4 = mod1(j+off[t,4,2],res); r4 = mod1(k+off[t,4,3],res)
        e1x = xg[p2,q2,r2,1]-xg[p1,q1,r1,1]; e1y = xg[p2,q2,r2,2]-xg[p1,q1,r1,2]; e1z = xg[p2,q2,r2,3]-xg[p1,q1,r1,3]
        e2x = xg[p3,q3,r3,1]-xg[p1,q1,r1,1]; e2y = xg[p3,q3,r3,2]-xg[p1,q1,r1,2]; e2z = xg[p3,q3,r3,3]-xg[p1,q1,r1,3]
        e3x = xg[p4,q4,r4,1]-xg[p1,q1,r1,1]; e3y = xg[p4,q4,r4,2]-xg[p1,q1,r1,2]; e3z = xg[p4,q4,r4,3]-xg[p1,q1,r1,3]
        V  = (e1x*(e2y*e3z-e2z*e3y) - e1y*(e2x*e3z-e2z*e3x) + e1z*(e2x*e3y-e2y*e3x)) / 6
        Vc = max(abs(V), floorvol)
        wT = (wg[p1,q1,r1]+wg[p2,q2,r2]+wg[p3,q3,r3]+wg[p4,q4,r4]) * oftype(V, 0.25)
        KernelAbstractions.@atomic D[1] += Float64(mT*wT/Vc)
        KernelAbstractions.@atomic Z[1] += Float64(mT*wT)
    end
end

# ── backward: recompute each tet; det cofactors → x̄_grid, weight mean → w̄ ──
@kernel function _sheet_reduce_bwd!(x̄, w̄, D̄, Z̄, @Const(xg), @Const(wg), @Const(off),
                                    res::Int, mT, floorvol)
    t, i, j, k = @index(Global, NTuple)
    @inbounds begin
        p1 = mod1(i+off[t,1,1],res); q1 = mod1(j+off[t,1,2],res); r1 = mod1(k+off[t,1,3],res)
        p2 = mod1(i+off[t,2,1],res); q2 = mod1(j+off[t,2,2],res); r2 = mod1(k+off[t,2,3],res)
        p3 = mod1(i+off[t,3,1],res); q3 = mod1(j+off[t,3,2],res); r3 = mod1(k+off[t,3,3],res)
        p4 = mod1(i+off[t,4,1],res); q4 = mod1(j+off[t,4,2],res); r4 = mod1(k+off[t,4,3],res)
        e1x = xg[p2,q2,r2,1]-xg[p1,q1,r1,1]; e1y = xg[p2,q2,r2,2]-xg[p1,q1,r1,2]; e1z = xg[p2,q2,r2,3]-xg[p1,q1,r1,3]
        e2x = xg[p3,q3,r3,1]-xg[p1,q1,r1,1]; e2y = xg[p3,q3,r3,2]-xg[p1,q1,r1,2]; e2z = xg[p3,q3,r3,3]-xg[p1,q1,r1,3]
        e3x = xg[p4,q4,r4,1]-xg[p1,q1,r1,1]; e3y = xg[p4,q4,r4,2]-xg[p1,q1,r1,2]; e3z = xg[p4,q4,r4,3]-xg[p1,q1,r1,3]
        V  = (e1x*(e2y*e3z-e2z*e3y) - e1y*(e2x*e3z-e2z*e3x) + e1z*(e2x*e3y-e2y*e3x)) / 6
        absV = abs(V); Vc = max(absV, floorvol)
        wT = (wg[p1,q1,r1]+wg[p2,q2,r2]+wg[p3,q3,r3]+wg[p4,q4,r4]) * oftype(V, 0.25)
        # cotangents:  V̄ = D̄·∂ρ/∂V (0 if floored);  w̄_T = D̄·∂ρ/∂w_T + Z̄·m_T
        dρdV = absV > floorvol ? (-mT*wT*sign(V)/(V*V)) : zero(V)
        V̄   = oftype(V, D̄) * dρdV
        w̄T  = oftype(V, D̄) * (mT/Vc) + oftype(V, Z̄) * mT
        s   = V̄ / 6
        # ȳ_i = (V̄/6)·cofactor_i  (cofactor₂=e₂×e₃, ₃=e₃×e₁, ₄=e₁×e₂, ₁=−Σ)
        g2x = s*(e2y*e3z-e2z*e3y); g2y = s*(e2z*e3x-e2x*e3z); g2z = s*(e2x*e3y-e2y*e3x)
        g3x = s*(e3y*e1z-e3z*e1y); g3y = s*(e3z*e1x-e3x*e1z); g3z = s*(e3x*e1y-e3y*e1x)
        g4x = s*(e1y*e2z-e1z*e2y); g4y = s*(e1z*e2x-e1x*e2z); g4z = s*(e1x*e2y-e1y*e2x)
        g1x = -(g2x+g3x+g4x); g1y = -(g2y+g3y+g4y); g1z = -(g2z+g3z+g4z)
        ww = w̄T * oftype(V, 0.25)
        KernelAbstractions.@atomic x̄[p1,q1,r1,1] += Float64(g1x); KernelAbstractions.@atomic x̄[p1,q1,r1,2] += Float64(g1y); KernelAbstractions.@atomic x̄[p1,q1,r1,3] += Float64(g1z)
        KernelAbstractions.@atomic x̄[p2,q2,r2,1] += Float64(g2x); KernelAbstractions.@atomic x̄[p2,q2,r2,2] += Float64(g2y); KernelAbstractions.@atomic x̄[p2,q2,r2,3] += Float64(g2z)
        KernelAbstractions.@atomic x̄[p3,q3,r3,1] += Float64(g3x); KernelAbstractions.@atomic x̄[p3,q3,r3,2] += Float64(g3y); KernelAbstractions.@atomic x̄[p3,q3,r3,3] += Float64(g3z)
        KernelAbstractions.@atomic x̄[p4,q4,r4,1] += Float64(g4x); KernelAbstractions.@atomic x̄[p4,q4,r4,2] += Float64(g4y); KernelAbstractions.@atomic x̄[p4,q4,r4,3] += Float64(g4z)
        KernelAbstractions.@atomic w̄[p1,q1,r1] += Float64(ww); KernelAbstractions.@atomic w̄[p2,q2,r2] += Float64(ww)
        KernelAbstractions.@atomic w̄[p3,q3,r3] += Float64(ww); KernelAbstractions.@atomic w̄[p4,q4,r4] += Float64(ww)
    end
end

"""
    sheet_tet_reduce(x_grid, w, res, boxsize; floor_frac=1e-3) -> (D, Z)

`D = Σ_T m_T·w_T/|V_T|` and `Z = Σ_T m_T·w_T` over the 6·res³ sheet tetrahedra.
Differentiable w.r.t. `x_grid::(res,res,res,3)` (through `V_T`) and `w::(res,res,res)`
(through `w_T`); `m_T = 1/6`, `V_floor = floor_frac·(boxsize/res)³/6`.
"""
function sheet_tet_reduce(x_grid::AbstractArray{T,4}, w::AbstractArray{T,3},
                          res::Int, boxsize::Real; floor_frac::Real=1e-3) where {T}
    backend = get_backend(x_grid)
    off = _offsets_on(x_grid)
    mT = T(1//6); floorvol = T(floor_frac) * (T(boxsize)/res)^3 / 6
    D = KernelAbstractions.zeros(backend, Float64, 1); Z = KernelAbstractions.zeros(backend, Float64, 1)
    _sheet_reduce_fwd!(backend)(D, Z, x_grid, w, off, res, mT, floorvol; ndrange=(6, res-1, res-1, res-1))
    synchronize(backend)
    return (Array(D)[1], Array(Z)[1])
end

function ChainRulesCore.rrule(::typeof(sheet_tet_reduce), x_grid::AbstractArray{T,4},
                              w::AbstractArray{T,3}, res::Int, boxsize::Real;
                              floor_frac::Real=1e-3) where {T}
    DZ = sheet_tet_reduce(x_grid, w, res, boxsize; floor_frac)
    off = _offsets_on(x_grid)
    mT = T(1//6); floorvol = T(floor_frac) * (T(boxsize)/res)^3 / 6
    function reduce_pullback(ΔDZ)
        D̄ = ΔDZ[1] isa ChainRulesCore.AbstractZero ? 0.0 : Float64(ΔDZ[1])
        Z̄ = ΔDZ[2] isa ChainRulesCore.AbstractZero ? 0.0 : Float64(ΔDZ[2])
        backend = get_backend(x_grid)
        x̄ = KernelAbstractions.zeros(backend, Float64, res, res, res, 3)
        w̄ = KernelAbstractions.zeros(backend, Float64, res, res, res)
        _sheet_reduce_bwd!(backend)(x̄, w̄, D̄, Z̄, x_grid, w, off, res, mT, floorvol; ndrange=(6, res-1, res-1, res-1))
        synchronize(backend)
        return (NoTangent(), T.(x̄), T.(w̄), NoTangent(), NoTangent())
    end
    return DZ, reduce_pullback
end

# Forward-only diagnostic: Σ_T V_T (signed) = box volume for any periodic deformation.
@kernel function _tet_vsum_kernel!(S, @Const(xg), @Const(off), res::Int)
    t, i, j, k = @index(Global, NTuple)
    @inbounds begin
        p1 = mod1(i+off[t,1,1],res); q1 = mod1(j+off[t,1,2],res); r1 = mod1(k+off[t,1,3],res)
        p2 = mod1(i+off[t,2,1],res); q2 = mod1(j+off[t,2,2],res); r2 = mod1(k+off[t,2,3],res)
        p3 = mod1(i+off[t,3,1],res); q3 = mod1(j+off[t,3,2],res); r3 = mod1(k+off[t,3,3],res)
        p4 = mod1(i+off[t,4,1],res); q4 = mod1(j+off[t,4,2],res); r4 = mod1(k+off[t,4,3],res)
        e1x = xg[p2,q2,r2,1]-xg[p1,q1,r1,1]; e1y = xg[p2,q2,r2,2]-xg[p1,q1,r1,2]; e1z = xg[p2,q2,r2,3]-xg[p1,q1,r1,3]
        e2x = xg[p3,q3,r3,1]-xg[p1,q1,r1,1]; e2y = xg[p3,q3,r3,2]-xg[p1,q1,r1,2]; e2z = xg[p3,q3,r3,3]-xg[p1,q1,r1,3]
        e3x = xg[p4,q4,r4,1]-xg[p1,q1,r1,1]; e3y = xg[p4,q4,r4,2]-xg[p1,q1,r1,2]; e3z = xg[p4,q4,r4,3]-xg[p1,q1,r1,3]
        V = (e1x*(e2y*e3z-e2z*e3y) - e1y*(e2x*e3z-e2z*e3x) + e1z*(e2x*e3y-e2y*e3x)) / 6
        KernelAbstractions.@atomic S[1] += Float64(V)
    end
end

"""    tet_volume_sum(x_grid, res) -> Σ_T V_T  (= box volume; det sanity check)"""
function tet_volume_sum(x_grid::AbstractArray{T,4}, res::Int) where {T}
    backend = get_backend(x_grid); off = _offsets_on(x_grid)
    S = KernelAbstractions.zeros(backend, Float64, 1)
    _tet_vsum_kernel!(backend)(S, x_grid, off, res; ndrange=(6, res-1, res-1, res-1))
    synchronize(backend)
    return Array(S)[1]
end
