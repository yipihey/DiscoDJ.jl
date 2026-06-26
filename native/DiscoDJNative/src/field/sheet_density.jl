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

# ── P2: detached point-location (fixed-galaxy cell list + per-tet barycentric query) ──
export build_cell_list, locate_points_in_sheet

"""
    build_cell_list(pts::(N,3), h) -> NamedTuple

Uniform chaining mesh on the FIXED query points (built once; the mesh deforms, the
galaxies don't).  Counting-sort into cells of size `h`: `perm[cell_start[c]:cell_start[c+1]-1]`
are the point indices in cell `c`.  CPU/host (detached — not on the AD tape).
"""
function build_cell_list(pts::AbstractMatrix{T}, h::Real) where {T}
    N = size(pts, 1); h = T(h)
    o1 = minimum(@view pts[:,1]); o2 = minimum(@view pts[:,2]); o3 = minimum(@view pts[:,3])
    m1 = maximum(@view pts[:,1]); m2 = maximum(@view pts[:,2]); m3 = maximum(@view pts[:,3])
    d1 = max(1, floor(Int,(m1-o1)/h)+1); d2 = max(1, floor(Int,(m2-o2)/h)+1); d3 = max(1, floor(Int,(m3-o3)/h)+1)
    nc = d1*d2*d3
    cid = Vector{Int}(undef, N); counts = zeros(Int, nc)
    @inbounds for g in 1:N
        cx = clamp(floor(Int,(pts[g,1]-o1)/h),0,d1-1); cy = clamp(floor(Int,(pts[g,2]-o2)/h),0,d2-1); cz = clamp(floor(Int,(pts[g,3]-o3)/h),0,d3-1)
        c = cx + d1*(cy + d2*cz) + 1; cid[g] = c; counts[c] += 1
    end
    cell_start = Vector{Int}(undef, nc+1); cell_start[1] = 1
    @inbounds for c in 1:nc; cell_start[c+1] = cell_start[c] + counts[c]; end
    perm = Vector{Int}(undef, N); fill_at = copy(cell_start)
    @inbounds for g in 1:N; c = cid[g]; perm[fill_at[c]] = g; fill_at[c] += 1; end
    return (o1=o1, o2=o2, o3=o3, h=h, d1=d1, d2=d2, d3=d3, cell_start=cell_start, perm=perm)
end

# per-tet: AABB → overlapping cells → barycentric test → atomic-increment each point's
# containment multiplicity (det reused for the barycentric solve, Cramer's rule).
@kernel function _locate_kernel!(mult, @Const(xg), @Const(off), @Const(pts), @Const(perm),
        @Const(cstart), o1, o2, o3, h, d1::Int, d2::Int, d3::Int, res::Int, eps)
    t, i, j, k = @index(Global, NTuple)
    @inbounds begin
        p1=i+off[t,1,1]; q1=j+off[t,1,2]; r1=k+off[t,1,3]
        p2=i+off[t,2,1]; q2=j+off[t,2,2]; r2=k+off[t,2,3]
        p3=i+off[t,3,1]; q3=j+off[t,3,2]; r3=k+off[t,3,3]
        p4=i+off[t,4,1]; q4=j+off[t,4,2]; r4=k+off[t,4,3]
        y1x=xg[p1,q1,r1,1]; y1y=xg[p1,q1,r1,2]; y1z=xg[p1,q1,r1,3]
        y2x=xg[p2,q2,r2,1]; y2y=xg[p2,q2,r2,2]; y2z=xg[p2,q2,r2,3]
        y3x=xg[p3,q3,r3,1]; y3y=xg[p3,q3,r3,2]; y3z=xg[p3,q3,r3,3]
        y4x=xg[p4,q4,r4,1]; y4y=xg[p4,q4,r4,2]; y4z=xg[p4,q4,r4,3]
        e1x=y2x-y1x; e1y=y2y-y1y; e1z=y2z-y1z
        e2x=y3x-y1x; e2y=y3y-y1y; e2z=y3z-y1z
        e3x=y4x-y1x; e3y=y4y-y1y; e3z=y4z-y1z
        c23x=e2y*e3z-e2z*e3y; c23y=e2z*e3x-e2x*e3z; c23z=e2x*e3y-e2y*e3x   # e₂×e₃
        detf = e1x*c23x + e1y*c23y + e1z*c23z                            # = 6 V_T
        if abs(detf) > eps
            inv = one(detf)/detf
            axmn=min(y1x,y2x,y3x,y4x); axmx=max(y1x,y2x,y3x,y4x)
            aymn=min(y1y,y2y,y3y,y4y); aymx=max(y1y,y2y,y3y,y4y)
            azmn=min(y1z,y2z,y3z,y4z); azmx=max(y1z,y2z,y3z,y4z)
            cxa=clamp(unsafe_trunc(Int,(axmn-o1)/h),0,d1-1); cxb=clamp(unsafe_trunc(Int,(axmx-o1)/h),0,d1-1)
            cya=clamp(unsafe_trunc(Int,(aymn-o2)/h),0,d2-1); cyb=clamp(unsafe_trunc(Int,(aymx-o2)/h),0,d2-1)
            cza=clamp(unsafe_trunc(Int,(azmn-o3)/h),0,d3-1); czb=clamp(unsafe_trunc(Int,(azmx-o3)/h),0,d3-1)
            for cz in cza:czb, cy in cya:cyb, cx in cxa:cxb
                c = cx + d1*(cy + d2*cz) + 1
                for idx in cstart[c]:(cstart[c+1]-1)
                    g = perm[idx]
                    dx=pts[g,1]-y1x; dy=pts[g,2]-y1y; dz=pts[g,3]-y1z
                    l2 = (dx*c23x + dy*c23y + dz*c23z)*inv
                    l3 = (e1x*(dy*e3z-dz*e3y) + e1y*(dz*e3x-dx*e3z) + e1z*(dx*e3y-dy*e3x))*inv
                    l4 = (e1x*(e2y*dz-e2z*dy) + e1y*(e2z*dx-e2x*dz) + e1z*(e2x*dy-e2y*dx))*inv
                    l1 = one(l2) - l2 - l3 - l4
                    tol = oftype(l2, eps)
                    if l1 >= -tol && l2 >= -tol && l3 >= -tol && l4 >= -tol
                        KernelAbstractions.@atomic mult[g] += Int32(1)
                    end
                end
            end
        end
    end
end

"""
    locate_points_in_sheet(x_grid, pts, cl, res; eps=1e-7) -> mult::(N,) Int32

For each query point, the number of sheet tetrahedra containing it (stream multiplicity:
1 single-stream, ≥3 in folded/multi-stream regions, 0 outside the sheet).  Detached.
`cl` = `build_cell_list(pts, h)`.  Reuses the per-tet edges + det.
"""
function locate_points_in_sheet(x_grid::AbstractArray{T,4}, pts::AbstractMatrix{T},
                                cl, res::Int; eps::Real=1e-7) where {T}
    backend = get_backend(x_grid); off = _offsets_on(x_grid)
    mv(x) = (y = similar(x_grid, eltype(x), size(x)); copyto!(y, x); y)   # to backend
    mult = KernelAbstractions.zeros(backend, Int32, size(pts,1))
    _locate_kernel!(backend)(mult, x_grid, off, mv(pts), mv(cl.perm), mv(cl.cell_start),
        T(cl.o1), T(cl.o2), T(cl.o3), T(cl.h), cl.d1, cl.d2, cl.d3, res, T(eps);
        ndrange=(6, res-1, res-1, res-1))
    synchronize(backend)
    return mult
end

# ── P3: the differentiable tet→galaxy deposit (the core primitive) ────────────
export sheet_density_at_points

# forward: per tet, ρ_T = m_T w_T/|V_T|; scatter to contained points; reduce Z = Σ m_T w_T
@kernel function _deposit_fwd!(ρg, Z, @Const(xg), @Const(wg), @Const(off), @Const(pts),
        @Const(perm), @Const(cstart), o1, o2, o3, h, d1::Int, d2::Int, d3::Int,
        res::Int, mT, floorvol, eps)
    t, i, j, k = @index(Global, NTuple)
    @inbounds begin
        p1=i+off[t,1,1];q1=j+off[t,1,2];r1=k+off[t,1,3]; p2=i+off[t,2,1];q2=j+off[t,2,2];r2=k+off[t,2,3]
        p3=i+off[t,3,1];q3=j+off[t,3,2];r3=k+off[t,3,3]; p4=i+off[t,4,1];q4=j+off[t,4,2];r4=k+off[t,4,3]
        y1x=xg[p1,q1,r1,1];y1y=xg[p1,q1,r1,2];y1z=xg[p1,q1,r1,3]; y2x=xg[p2,q2,r2,1];y2y=xg[p2,q2,r2,2];y2z=xg[p2,q2,r2,3]
        y3x=xg[p3,q3,r3,1];y3y=xg[p3,q3,r3,2];y3z=xg[p3,q3,r3,3]; y4x=xg[p4,q4,r4,1];y4y=xg[p4,q4,r4,2];y4z=xg[p4,q4,r4,3]
        e1x=y2x-y1x;e1y=y2y-y1y;e1z=y2z-y1z; e2x=y3x-y1x;e2y=y3y-y1y;e2z=y3z-y1z; e3x=y4x-y1x;e3y=y4y-y1y;e3z=y4z-y1z
        c23x=e2y*e3z-e2z*e3y;c23y=e2z*e3x-e2x*e3z;c23z=e2x*e3y-e2y*e3x
        detf = e1x*c23x+e1y*c23y+e1z*c23z
        wT = (wg[p1,q1,r1]+wg[p2,q2,r2]+wg[p3,q3,r3]+wg[p4,q4,r4])*oftype(detf,0.25)
        KernelAbstractions.@atomic Z[1] += Float64(mT*wT)
        if abs(detf) > eps
            ρT = mT*wT/max(abs(detf)/6, floorvol); inv = one(detf)/detf
            axmn=min(y1x,y2x,y3x,y4x);axmx=max(y1x,y2x,y3x,y4x); aymn=min(y1y,y2y,y3y,y4y);aymx=max(y1y,y2y,y3y,y4y); azmn=min(y1z,y2z,y3z,y4z);azmx=max(y1z,y2z,y3z,y4z)
            cxa=clamp(unsafe_trunc(Int,(axmn-o1)/h),0,d1-1);cxb=clamp(unsafe_trunc(Int,(axmx-o1)/h),0,d1-1)
            cya=clamp(unsafe_trunc(Int,(aymn-o2)/h),0,d2-1);cyb=clamp(unsafe_trunc(Int,(aymx-o2)/h),0,d2-1)
            cza=clamp(unsafe_trunc(Int,(azmn-o3)/h),0,d3-1);czb=clamp(unsafe_trunc(Int,(azmx-o3)/h),0,d3-1)
            for cz in cza:czb, cy in cya:cyb, cx in cxa:cxb
                c = cx + d1*(cy + d2*cz) + 1
                for idx in cstart[c]:(cstart[c+1]-1)
                    g = perm[idx]; dx=pts[g,1]-y1x;dy=pts[g,2]-y1y;dz=pts[g,3]-y1z
                    l2=(dx*c23x+dy*c23y+dz*c23z)*inv
                    l3=(e1x*(dy*e3z-dz*e3y)+e1y*(dz*e3x-dx*e3z)+e1z*(dx*e3y-dy*e3x))*inv
                    l4=(e1x*(e2y*dz-e2z*dy)+e1y*(e2z*dx-e2x*dz)+e1z*(e2x*dy-e2y*dx))*inv
                    l1=one(l2)-l2-l3-l4; tol=oftype(l2,eps)
                    if l1>=-tol&&l2>=-tol&&l3>=-tol&&l4>=-tol
                        KernelAbstractions.@atomic ρg[g] += Float64(ρT)
                    end
                end
            end
        end
    end
end

# backward: per tet, gather ρ̄_T = Σ_{g∈T} ρ̄g[g]; then ρ_T's V/w derivatives → cofactor
# adjoint → x̄_grid, weight-mean adjoint → w̄ (recompute-in-backward; locator re-run).
@kernel function _deposit_bwd!(x̄, w̄, @Const(ρ̄g), Z̄, @Const(xg), @Const(wg), @Const(off),
        @Const(pts), @Const(perm), @Const(cstart), o1, o2, o3, h, d1::Int, d2::Int, d3::Int,
        res::Int, mT, floorvol, eps)
    t, i, j, k = @index(Global, NTuple)
    @inbounds begin
        p1=i+off[t,1,1];q1=j+off[t,1,2];r1=k+off[t,1,3]; p2=i+off[t,2,1];q2=j+off[t,2,2];r2=k+off[t,2,3]
        p3=i+off[t,3,1];q3=j+off[t,3,2];r3=k+off[t,3,3]; p4=i+off[t,4,1];q4=j+off[t,4,2];r4=k+off[t,4,3]
        y1x=xg[p1,q1,r1,1];y1y=xg[p1,q1,r1,2];y1z=xg[p1,q1,r1,3]; y2x=xg[p2,q2,r2,1];y2y=xg[p2,q2,r2,2];y2z=xg[p2,q2,r2,3]
        y3x=xg[p3,q3,r3,1];y3y=xg[p3,q3,r3,2];y3z=xg[p3,q3,r3,3]; y4x=xg[p4,q4,r4,1];y4y=xg[p4,q4,r4,2];y4z=xg[p4,q4,r4,3]
        e1x=y2x-y1x;e1y=y2y-y1y;e1z=y2z-y1z; e2x=y3x-y1x;e2y=y3y-y1y;e2z=y3z-y1z; e3x=y4x-y1x;e3y=y4y-y1y;e3z=y4z-y1z
        c23x=e2y*e3z-e2z*e3y;c23y=e2z*e3x-e2x*e3z;c23z=e2x*e3y-e2y*e3x
        detf = e1x*c23x+e1y*c23y+e1z*c23z; V = detf/6; absV = abs(V); Vc = max(absV, floorvol)
        wT = (wg[p1,q1,r1]+wg[p2,q2,r2]+wg[p3,q3,r3]+wg[p4,q4,r4])*oftype(detf,0.25)
        rbar = zero(Float64)
        if abs(detf) > eps
            inv = one(detf)/detf
            axmn=min(y1x,y2x,y3x,y4x);axmx=max(y1x,y2x,y3x,y4x); aymn=min(y1y,y2y,y3y,y4y);aymx=max(y1y,y2y,y3y,y4y); azmn=min(y1z,y2z,y3z,y4z);azmx=max(y1z,y2z,y3z,y4z)
            cxa=clamp(unsafe_trunc(Int,(axmn-o1)/h),0,d1-1);cxb=clamp(unsafe_trunc(Int,(axmx-o1)/h),0,d1-1)
            cya=clamp(unsafe_trunc(Int,(aymn-o2)/h),0,d2-1);cyb=clamp(unsafe_trunc(Int,(aymx-o2)/h),0,d2-1)
            cza=clamp(unsafe_trunc(Int,(azmn-o3)/h),0,d3-1);czb=clamp(unsafe_trunc(Int,(azmx-o3)/h),0,d3-1)
            for cz in cza:czb, cy in cya:cyb, cx in cxa:cxb
                c = cx + d1*(cy + d2*cz) + 1
                for idx in cstart[c]:(cstart[c+1]-1)
                    g = perm[idx]; dx=pts[g,1]-y1x;dy=pts[g,2]-y1y;dz=pts[g,3]-y1z
                    l2=(dx*c23x+dy*c23y+dz*c23z)*inv
                    l3=(e1x*(dy*e3z-dz*e3y)+e1y*(dz*e3x-dx*e3z)+e1z*(dx*e3y-dy*e3x))*inv
                    l4=(e1x*(e2y*dz-e2z*dy)+e1y*(e2z*dx-e2x*dz)+e1z*(e2x*dy-e2y*dx))*inv
                    l1=one(l2)-l2-l3-l4; tol=oftype(l2,eps)
                    (l1>=-tol&&l2>=-tol&&l3>=-tol&&l4>=-tol) && (rbar += ρ̄g[g])
                end
            end
        end
        ρ̄T = oftype(V, rbar)
        dρdV = absV > floorvol ? (-mT*wT*sign(V)/(V*V)) : zero(V)
        V̄  = ρ̄T * dρdV
        w̄T = ρ̄T * (mT/Vc) + oftype(V, Z̄) * mT
        s = V̄ / 6
        g2x=s*c23x; g2y=s*c23y; g2z=s*c23z
        g3x=s*(e3y*e1z-e3z*e1y); g3y=s*(e3z*e1x-e3x*e1z); g3z=s*(e3x*e1y-e3y*e1x)
        g4x=s*(e1y*e2z-e1z*e2y); g4y=s*(e1z*e2x-e1x*e2z); g4z=s*(e1x*e2y-e1y*e2x)
        g1x=-(g2x+g3x+g4x); g1y=-(g2y+g3y+g4y); g1z=-(g2z+g3z+g4z); ww=w̄T*oftype(V,0.25)
        KernelAbstractions.@atomic x̄[p1,q1,r1,1]+=Float64(g1x); KernelAbstractions.@atomic x̄[p1,q1,r1,2]+=Float64(g1y); KernelAbstractions.@atomic x̄[p1,q1,r1,3]+=Float64(g1z)
        KernelAbstractions.@atomic x̄[p2,q2,r2,1]+=Float64(g2x); KernelAbstractions.@atomic x̄[p2,q2,r2,2]+=Float64(g2y); KernelAbstractions.@atomic x̄[p2,q2,r2,3]+=Float64(g2z)
        KernelAbstractions.@atomic x̄[p3,q3,r3,1]+=Float64(g3x); KernelAbstractions.@atomic x̄[p3,q3,r3,2]+=Float64(g3y); KernelAbstractions.@atomic x̄[p3,q3,r3,3]+=Float64(g3z)
        KernelAbstractions.@atomic x̄[p4,q4,r4,1]+=Float64(g4x); KernelAbstractions.@atomic x̄[p4,q4,r4,2]+=Float64(g4y); KernelAbstractions.@atomic x̄[p4,q4,r4,3]+=Float64(g4z)
        KernelAbstractions.@atomic w̄[p1,q1,r1]+=Float64(ww); KernelAbstractions.@atomic w̄[p2,q2,r2]+=Float64(ww); KernelAbstractions.@atomic w̄[p3,q3,r3]+=Float64(ww); KernelAbstractions.@atomic w̄[p4,q4,r4]+=Float64(ww)
    end
end

"""
    sheet_density_at_points(x_grid, w, pts, cl, res, boxsize; floor_frac=1e-3, eps=1e-7)
        -> (ρ_g::(N,), Z)

AHK galaxy density at each query point on the deformed sheet — `ρ_g[g] = Σ_{T∋g} m_T w_T/|V_T|`
(piecewise-constant; sum over containing tets ⇒ multi-streaming) — and the normalisation
`Z = Σ_T m_T w_T`.  Differentiable w.r.t. `x_grid` (det cofactors) and `w` (weight mean);
the tet→point assignment is detached and recomputed in the backward.  Hand-written `rrule`.
"""
function sheet_density_at_points(x_grid::AbstractArray{T,4}, w::AbstractArray{T,3},
        pts::AbstractMatrix{T}, cl, res::Int, boxsize::Real;
        floor_frac::Real=1e-3, eps::Real=1e-7) where {T}
    backend = get_backend(x_grid); off = _offsets_on(x_grid)
    mv(x) = (y = similar(x_grid, eltype(x), size(x)); copyto!(y, x); y)
    mT = T(1//6); floorvol = T(floor_frac)*(T(boxsize)/res)^3/6
    ρg = KernelAbstractions.zeros(backend, Float64, size(pts,1)); Z = KernelAbstractions.zeros(backend, Float64, 1)
    _deposit_fwd!(backend)(ρg, Z, x_grid, w, off, mv(pts), mv(cl.perm), mv(cl.cell_start),
        T(cl.o1),T(cl.o2),T(cl.o3),T(cl.h),cl.d1,cl.d2,cl.d3, res, mT, floorvol, T(eps);
        ndrange=(6,res-1,res-1,res-1))
    synchronize(backend)
    return (ρg, Array(Z)[1])
end

function ChainRulesCore.rrule(::typeof(sheet_density_at_points), x_grid::AbstractArray{T,4},
        w::AbstractArray{T,3}, pts::AbstractMatrix{T}, cl, res::Int, boxsize::Real;
        floor_frac::Real=1e-3, eps::Real=1e-7) where {T}
    out = sheet_density_at_points(x_grid, w, pts, cl, res, boxsize; floor_frac, eps)
    backend = get_backend(x_grid); off = _offsets_on(x_grid)
    mv(x) = (y = similar(x_grid, eltype(x), size(x)); copyto!(y, x); y)
    ptsb = mv(pts); permb = mv(cl.perm); cstartb = mv(cl.cell_start)
    mT = T(1//6); floorvol = T(floor_frac)*(T(boxsize)/res)^3/6
    function deposit_pullback(Δ)
        ρ̄g = Δ[1] isa ChainRulesCore.AbstractZero ? KernelAbstractions.zeros(backend, Float64, size(pts,1)) :
             (y = KernelAbstractions.zeros(backend, Float64, size(pts,1)); copyto!(y, Float64.(unthunk(Δ[1]))); y)
        Z̄ = Δ[2] isa ChainRulesCore.AbstractZero ? 0.0 : Float64(Δ[2])
        x̄ = KernelAbstractions.zeros(backend, Float64, res, res, res, 3)
        w̄ = KernelAbstractions.zeros(backend, Float64, res, res, res)
        _deposit_bwd!(backend)(x̄, w̄, ρ̄g, Z̄, x_grid, w, off, ptsb, permb, cstartb,
            T(cl.o1),T(cl.o2),T(cl.o3),T(cl.h),cl.d1,cl.d2,cl.d3, res, mT, floorvol, T(eps);
            ndrange=(6,res-1,res-1,res-1))
        synchronize(backend)
        return (NoTangent(), T.(x̄), T.(w̄), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end
    return out, deposit_pullback
end

# ── P6: C⁰ nodal-averaged density (continuous across tet faces) ────────────────
export nodal_density, interp_sheet_at_points

# tet→vertex reduction: N_v=Σ_{T∋v} m_T w_T, D_v=Σ_{T∋v} max(|V_T|,floor), Z=Σ_T m_T w_T
@kernel function _nodal_fwd!(Nv, Dv, Z, @Const(xg), @Const(wg), @Const(off), res::Int, mT, floorvol)
    t, i, j, k = @index(Global, NTuple)
    @inbounds begin
        p1=i+off[t,1,1];q1=j+off[t,1,2];r1=k+off[t,1,3]; p2=i+off[t,2,1];q2=j+off[t,2,2];r2=k+off[t,2,3]
        p3=i+off[t,3,1];q3=j+off[t,3,2];r3=k+off[t,3,3]; p4=i+off[t,4,1];q4=j+off[t,4,2];r4=k+off[t,4,3]
        e1x=xg[p2,q2,r2,1]-xg[p1,q1,r1,1];e1y=xg[p2,q2,r2,2]-xg[p1,q1,r1,2];e1z=xg[p2,q2,r2,3]-xg[p1,q1,r1,3]
        e2x=xg[p3,q3,r3,1]-xg[p1,q1,r1,1];e2y=xg[p3,q3,r3,2]-xg[p1,q1,r1,2];e2z=xg[p3,q3,r3,3]-xg[p1,q1,r1,3]
        e3x=xg[p4,q4,r4,1]-xg[p1,q1,r1,1];e3y=xg[p4,q4,r4,2]-xg[p1,q1,r1,2];e3z=xg[p4,q4,r4,3]-xg[p1,q1,r1,3]
        detf=e1x*(e2y*e3z-e2z*e3y)-e1y*(e2x*e3z-e2z*e3x)+e1z*(e2x*e3y-e2y*e3x)
        Vc=max(abs(detf)/6,floorvol)
        wT=(wg[p1,q1,r1]+wg[p2,q2,r2]+wg[p3,q3,r3]+wg[p4,q4,r4])*oftype(detf,0.25); mw=Float64(mT*wT)
        KernelAbstractions.@atomic Z[1]+=mw
        KernelAbstractions.@atomic Nv[p1,q1,r1]+=mw; KernelAbstractions.@atomic Nv[p2,q2,r2]+=mw; KernelAbstractions.@atomic Nv[p3,q3,r3]+=mw; KernelAbstractions.@atomic Nv[p4,q4,r4]+=mw
        dv=Float64(Vc)
        KernelAbstractions.@atomic Dv[p1,q1,r1]+=dv; KernelAbstractions.@atomic Dv[p2,q2,r2]+=dv; KernelAbstractions.@atomic Dv[p3,q3,r3]+=dv; KernelAbstractions.@atomic Dv[p4,q4,r4]+=dv
    end
end

# backward: gather N̄_v,D̄_v over a tet → w̄_T, |V̄_T| → weight + cofactor adjoints
@kernel function _nodal_bwd!(x̄, w̄, @Const(N̄v), @Const(D̄v), Z̄, @Const(xg), @Const(off), res::Int, mT, floorvol)
    t, i, j, k = @index(Global, NTuple)
    @inbounds begin
        p1=i+off[t,1,1];q1=j+off[t,1,2];r1=k+off[t,1,3]; p2=i+off[t,2,1];q2=j+off[t,2,2];r2=k+off[t,2,3]
        p3=i+off[t,3,1];q3=j+off[t,3,2];r3=k+off[t,3,3]; p4=i+off[t,4,1];q4=j+off[t,4,2];r4=k+off[t,4,3]
        e1x=xg[p2,q2,r2,1]-xg[p1,q1,r1,1];e1y=xg[p2,q2,r2,2]-xg[p1,q1,r1,2];e1z=xg[p2,q2,r2,3]-xg[p1,q1,r1,3]
        e2x=xg[p3,q3,r3,1]-xg[p1,q1,r1,1];e2y=xg[p3,q3,r3,2]-xg[p1,q1,r1,2];e2z=xg[p3,q3,r3,3]-xg[p1,q1,r1,3]
        e3x=xg[p4,q4,r4,1]-xg[p1,q1,r1,1];e3y=xg[p4,q4,r4,2]-xg[p1,q1,r1,2];e3z=xg[p4,q4,r4,3]-xg[p1,q1,r1,3]
        c23x=e2y*e3z-e2z*e3y;c23y=e2z*e3x-e2x*e3z;c23z=e2x*e3y-e2y*e3x
        detf=e1x*c23x+e1y*c23y+e1z*c23z; V=detf/6
        sumN̄=N̄v[p1,q1,r1]+N̄v[p2,q2,r2]+N̄v[p3,q3,r3]+N̄v[p4,q4,r4]
        sumD̄=D̄v[p1,q1,r1]+D̄v[p2,q2,r2]+D̄v[p3,q3,r3]+D̄v[p4,q4,r4]
        w̄T = oftype(V, mT*sumN̄ + Z̄*mT)
        V̄ = abs(V) > floorvol ? oftype(V, sign(V)*sumD̄) : zero(V)
        s = V̄/6
        g2x=s*c23x;g2y=s*c23y;g2z=s*c23z
        g3x=s*(e3y*e1z-e3z*e1y);g3y=s*(e3z*e1x-e3x*e1z);g3z=s*(e3x*e1y-e3y*e1x)
        g4x=s*(e1y*e2z-e1z*e2y);g4y=s*(e1z*e2x-e1x*e2z);g4z=s*(e1x*e2y-e1y*e2x)
        g1x=-(g2x+g3x+g4x);g1y=-(g2y+g3y+g4y);g1z=-(g2z+g3z+g4z); ww=w̄T*oftype(V,0.25)
        KernelAbstractions.@atomic x̄[p1,q1,r1,1]+=Float64(g1x);KernelAbstractions.@atomic x̄[p1,q1,r1,2]+=Float64(g1y);KernelAbstractions.@atomic x̄[p1,q1,r1,3]+=Float64(g1z)
        KernelAbstractions.@atomic x̄[p2,q2,r2,1]+=Float64(g2x);KernelAbstractions.@atomic x̄[p2,q2,r2,2]+=Float64(g2y);KernelAbstractions.@atomic x̄[p2,q2,r2,3]+=Float64(g2z)
        KernelAbstractions.@atomic x̄[p3,q3,r3,1]+=Float64(g3x);KernelAbstractions.@atomic x̄[p3,q3,r3,2]+=Float64(g3y);KernelAbstractions.@atomic x̄[p3,q3,r3,3]+=Float64(g3z)
        KernelAbstractions.@atomic x̄[p4,q4,r4,1]+=Float64(g4x);KernelAbstractions.@atomic x̄[p4,q4,r4,2]+=Float64(g4y);KernelAbstractions.@atomic x̄[p4,q4,r4,3]+=Float64(g4z)
        KernelAbstractions.@atomic w̄[p1,q1,r1]+=Float64(ww);KernelAbstractions.@atomic w̄[p2,q2,r2]+=Float64(ww);KernelAbstractions.@atomic w̄[p3,q3,r3]+=Float64(ww);KernelAbstractions.@atomic w̄[p4,q4,r4]+=Float64(ww)
    end
end

"""    nodal_density(x_grid, w, res, boxsize; floor_frac=1e-3) -> (ρ_v::(res,res,res), Z)

Per-vertex AHK density `ρ_v = Σ_{T∋v} m_T w_T / Σ_{T∋v} max(|V_T|,V_floor)` (volume-weighted
mean of incident tets) and `Z = Σ_T m_T w_T`.  The C⁰ field's nodes; differentiable."""
function nodal_density(x_grid::AbstractArray{T,4}, w::AbstractArray{T,3}, res::Int, boxsize::Real;
                       floor_frac::Real=1e-3) where {T}
    backend = get_backend(x_grid); off = _offsets_on(x_grid)
    mT = T(1//6); floorvol = T(floor_frac)*(T(boxsize)/res)^3/6
    Nv = KernelAbstractions.zeros(backend, Float64, res, res, res); Dv = KernelAbstractions.zeros(backend, Float64, res, res, res)
    Z = KernelAbstractions.zeros(backend, Float64, 1)
    _nodal_fwd!(backend)(Nv, Dv, Z, x_grid, w, off, res, mT, floorvol; ndrange=(6,res-1,res-1,res-1))
    synchronize(backend)
    Dc = max.(Dv, Float64(floorvol)); ρv = T.(Nv ./ Dc)
    return (ρv, Array(Z)[1])
end

function ChainRulesCore.rrule(::typeof(nodal_density), x_grid::AbstractArray{T,4},
                              w::AbstractArray{T,3}, res::Int, boxsize::Real; floor_frac::Real=1e-3) where {T}
    backend = get_backend(x_grid); off = _offsets_on(x_grid)
    mT = T(1//6); floorvol = T(floor_frac)*(T(boxsize)/res)^3/6
    Nv = KernelAbstractions.zeros(backend, Float64, res, res, res); Dv = KernelAbstractions.zeros(backend, Float64, res, res, res)
    Z = KernelAbstractions.zeros(backend, Float64, 1)
    _nodal_fwd!(backend)(Nv, Dv, Z, x_grid, w, off, res, mT, floorvol; ndrange=(6,res-1,res-1,res-1))
    synchronize(backend)
    Dc = max.(Dv, Float64(floorvol)); ρv = T.(Nv ./ Dc); Zv = Array(Z)[1]
    function nodal_pullback(Δ)
        ρ̄v = Δ[1] isa ChainRulesCore.AbstractZero ? KernelAbstractions.zeros(backend, Float64, res,res,res) :
             (y=KernelAbstractions.zeros(backend,Float64,res,res,res); copyto!(y, Float64.(unthunk(Δ[1]))); y)
        Z̄ = Δ[2] isa ChainRulesCore.AbstractZero ? 0.0 : Float64(Δ[2])
        N̄v = ρ̄v ./ Dc
        D̄v = @. -ρ̄v * Nv / (Dc*Dc) * (Dv > Float64(floorvol))
        x̄ = KernelAbstractions.zeros(backend, Float64, res,res,res,3); w̄ = KernelAbstractions.zeros(backend, Float64, res,res,res)
        _nodal_bwd!(backend)(x̄, w̄, N̄v, D̄v, Z̄, x_grid, off, res, mT, floorvol; ndrange=(6,res-1,res-1,res-1))
        synchronize(backend)
        return (NoTangent(), T.(x̄), T.(w̄), NoTangent(), NoTangent())
    end
    return (ρv, Zv), nodal_pullback
end

# barycentric interpolation of vertex densities at the query points (C⁰)
@kernel function _interp_fwd!(ρg, @Const(xg), @Const(ρv), @Const(off), @Const(pts), @Const(perm),
        @Const(cstart), o1,o2,o3,h,d1::Int,d2::Int,d3::Int, res::Int, eps)
    t,i,j,k = @index(Global, NTuple)
    @inbounds begin
        p1=i+off[t,1,1];q1=j+off[t,1,2];r1=k+off[t,1,3]; p2=i+off[t,2,1];q2=j+off[t,2,2];r2=k+off[t,2,3]
        p3=i+off[t,3,1];q3=j+off[t,3,2];r3=k+off[t,3,3]; p4=i+off[t,4,1];q4=j+off[t,4,2];r4=k+off[t,4,3]
        y1x=xg[p1,q1,r1,1];y1y=xg[p1,q1,r1,2];y1z=xg[p1,q1,r1,3]; y2x=xg[p2,q2,r2,1];y2y=xg[p2,q2,r2,2];y2z=xg[p2,q2,r2,3]
        y3x=xg[p3,q3,r3,1];y3y=xg[p3,q3,r3,2];y3z=xg[p3,q3,r3,3]; y4x=xg[p4,q4,r4,1];y4y=xg[p4,q4,r4,2];y4z=xg[p4,q4,r4,3]
        e1x=y2x-y1x;e1y=y2y-y1y;e1z=y2z-y1z;e2x=y3x-y1x;e2y=y3y-y1y;e2z=y3z-y1z;e3x=y4x-y1x;e3y=y4y-y1y;e3z=y4z-y1z
        c23x=e2y*e3z-e2z*e3y;c23y=e2z*e3x-e2x*e3z;c23z=e2x*e3y-e2y*e3x; detf=e1x*c23x+e1y*c23y+e1z*c23z
        if abs(detf)>eps
            inv=one(detf)/detf; ρ1=ρv[p1,q1,r1];ρ2=ρv[p2,q2,r2];ρ3=ρv[p3,q3,r3];ρ4=ρv[p4,q4,r4]
            axmn=min(y1x,y2x,y3x,y4x);axmx=max(y1x,y2x,y3x,y4x);aymn=min(y1y,y2y,y3y,y4y);aymx=max(y1y,y2y,y3y,y4y);azmn=min(y1z,y2z,y3z,y4z);azmx=max(y1z,y2z,y3z,y4z)
            cxa=clamp(unsafe_trunc(Int,(axmn-o1)/h),0,d1-1);cxb=clamp(unsafe_trunc(Int,(axmx-o1)/h),0,d1-1)
            cya=clamp(unsafe_trunc(Int,(aymn-o2)/h),0,d2-1);cyb=clamp(unsafe_trunc(Int,(aymx-o2)/h),0,d2-1)
            cza=clamp(unsafe_trunc(Int,(azmn-o3)/h),0,d3-1);czb=clamp(unsafe_trunc(Int,(azmx-o3)/h),0,d3-1)
            for cz in cza:czb, cy in cya:cyb, cx in cxa:cxb
                c=cx+d1*(cy+d2*cz)+1
                for idx in cstart[c]:(cstart[c+1]-1)
                    g=perm[idx]; dx=pts[g,1]-y1x;dy=pts[g,2]-y1y;dz=pts[g,3]-y1z
                    l2=(dx*c23x+dy*c23y+dz*c23z)*inv
                    l3=(e1x*(dy*e3z-dz*e3y)+e1y*(dz*e3x-dx*e3z)+e1z*(dx*e3y-dy*e3x))*inv
                    l4=(e1x*(e2y*dz-e2z*dy)+e1y*(e2z*dx-e2x*dz)+e1z*(e2x*dy-e2y*dx))*inv
                    l1=one(l2)-l2-l3-l4; tol=oftype(l2,eps)
                    if l1>=-tol&&l2>=-tol&&l3>=-tol&&l4>=-tol
                        KernelAbstractions.@atomic ρg[g]+=Float64(l1*ρ1+l2*ρ2+l3*ρ3+l4*ρ4)
                    end
                end
            end
        end
    end
end

# backward: ρ̄_g → ρ̄_v += λ_i ρ̄_g (gather); x̄ += −ρ̄_g λ_j (∇ρ)_T   (∂λ_i/∂y_j = −λ_j ∇λ_i)
@kernel function _interp_bwd!(x̄, ρ̄v, @Const(ρ̄g), @Const(xg), @Const(ρv), @Const(off), @Const(pts),
        @Const(perm), @Const(cstart), o1,o2,o3,h,d1::Int,d2::Int,d3::Int, res::Int, eps)
    t,i,j,k = @index(Global, NTuple)
    @inbounds begin
        p1=i+off[t,1,1];q1=j+off[t,1,2];r1=k+off[t,1,3]; p2=i+off[t,2,1];q2=j+off[t,2,2];r2=k+off[t,2,3]
        p3=i+off[t,3,1];q3=j+off[t,3,2];r3=k+off[t,3,3]; p4=i+off[t,4,1];q4=j+off[t,4,2];r4=k+off[t,4,3]
        y1x=xg[p1,q1,r1,1];y1y=xg[p1,q1,r1,2];y1z=xg[p1,q1,r1,3]; y2x=xg[p2,q2,r2,1];y2y=xg[p2,q2,r2,2];y2z=xg[p2,q2,r2,3]
        y3x=xg[p3,q3,r3,1];y3y=xg[p3,q3,r3,2];y3z=xg[p3,q3,r3,3]; y4x=xg[p4,q4,r4,1];y4y=xg[p4,q4,r4,2];y4z=xg[p4,q4,r4,3]
        e1x=y2x-y1x;e1y=y2y-y1y;e1z=y2z-y1z;e2x=y3x-y1x;e2y=y3y-y1y;e2z=y3z-y1z;e3x=y4x-y1x;e3y=y4y-y1y;e3z=y4z-y1z
        c23x=e2y*e3z-e2z*e3y;c23y=e2z*e3x-e2x*e3z;c23z=e2x*e3y-e2y*e3x; detf=e1x*c23x+e1y*c23y+e1z*c23z
        if abs(detf)>eps
            inv=one(detf)/detf; ρ1=ρv[p1,q1,r1];ρ2=ρv[p2,q2,r2];ρ3=ρv[p3,q3,r3];ρ4=ρv[p4,q4,r4]
            c31x=e3y*e1z-e3z*e1y;c31y=e3z*e1x-e3x*e1z;c31z=e3x*e1y-e3y*e1x
            c12x=e1y*e2z-e1z*e2y;c12y=e1z*e2x-e1x*e2z;c12z=e1x*e2y-e1y*e2x
            gρx=((ρ2-ρ1)*c23x+(ρ3-ρ1)*c31x+(ρ4-ρ1)*c12x)*inv   # (∇ρ)_T (constant per tet)
            gρy=((ρ2-ρ1)*c23y+(ρ3-ρ1)*c31y+(ρ4-ρ1)*c12y)*inv
            gρz=((ρ2-ρ1)*c23z+(ρ3-ρ1)*c31z+(ρ4-ρ1)*c12z)*inv
            axmn=min(y1x,y2x,y3x,y4x);axmx=max(y1x,y2x,y3x,y4x);aymn=min(y1y,y2y,y3y,y4y);aymx=max(y1y,y2y,y3y,y4y);azmn=min(y1z,y2z,y3z,y4z);azmx=max(y1z,y2z,y3z,y4z)
            cxa=clamp(unsafe_trunc(Int,(axmn-o1)/h),0,d1-1);cxb=clamp(unsafe_trunc(Int,(axmx-o1)/h),0,d1-1)
            cya=clamp(unsafe_trunc(Int,(aymn-o2)/h),0,d2-1);cyb=clamp(unsafe_trunc(Int,(aymx-o2)/h),0,d2-1)
            cza=clamp(unsafe_trunc(Int,(azmn-o3)/h),0,d3-1);czb=clamp(unsafe_trunc(Int,(azmx-o3)/h),0,d3-1)
            for cz in cza:czb, cy in cya:cyb, cx in cxa:cxb
                c=cx+d1*(cy+d2*cz)+1
                for idx in cstart[c]:(cstart[c+1]-1)
                    g=perm[idx]; dx=pts[g,1]-y1x;dy=pts[g,2]-y1y;dz=pts[g,3]-y1z
                    l2=(dx*c23x+dy*c23y+dz*c23z)*inv
                    l3=(e1x*(dy*e3z-dz*e3y)+e1y*(dz*e3x-dx*e3z)+e1z*(dx*e3y-dy*e3x))*inv
                    l4=(e1x*(e2y*dz-e2z*dy)+e1y*(e2z*dx-e2x*dz)+e1z*(e2x*dy-e2y*dx))*inv
                    l1=one(l2)-l2-l3-l4; tol=oftype(l2,eps)
                    if l1>=-tol&&l2>=-tol&&l3>=-tol&&l4>=-tol
                        rb=ρ̄g[g]
                        KernelAbstractions.@atomic ρ̄v[p1,q1,r1]+=Float64(l1*rb);KernelAbstractions.@atomic ρ̄v[p2,q2,r2]+=Float64(l2*rb);KernelAbstractions.@atomic ρ̄v[p3,q3,r3]+=Float64(l3*rb);KernelAbstractions.@atomic ρ̄v[p4,q4,r4]+=Float64(l4*rb)
                        b1=-rb*l1;b2=-rb*l2;b3=-rb*l3;b4=-rb*l4   # ȳ_j = −ρ̄_g λ_j (∇ρ)_T
                        KernelAbstractions.@atomic x̄[p1,q1,r1,1]+=Float64(b1*gρx);KernelAbstractions.@atomic x̄[p1,q1,r1,2]+=Float64(b1*gρy);KernelAbstractions.@atomic x̄[p1,q1,r1,3]+=Float64(b1*gρz)
                        KernelAbstractions.@atomic x̄[p2,q2,r2,1]+=Float64(b2*gρx);KernelAbstractions.@atomic x̄[p2,q2,r2,2]+=Float64(b2*gρy);KernelAbstractions.@atomic x̄[p2,q2,r2,3]+=Float64(b2*gρz)
                        KernelAbstractions.@atomic x̄[p3,q3,r3,1]+=Float64(b3*gρx);KernelAbstractions.@atomic x̄[p3,q3,r3,2]+=Float64(b3*gρy);KernelAbstractions.@atomic x̄[p3,q3,r3,3]+=Float64(b3*gρz)
                        KernelAbstractions.@atomic x̄[p4,q4,r4,1]+=Float64(b4*gρx);KernelAbstractions.@atomic x̄[p4,q4,r4,2]+=Float64(b4*gρy);KernelAbstractions.@atomic x̄[p4,q4,r4,3]+=Float64(b4*gρz)
                    end
                end
            end
        end
    end
end

"""    interp_sheet_at_points(x_grid, ρ_v, pts, cl, res; eps=1e-7) -> ρ_g

C⁰ density at the query points: `ρ_g = Σ_{T∋g} Σ_i λ_i ρ_{v_i}` (barycentric interpolation
of the vertex densities; sum over containing tets).  Differentiable w.r.t. `x_grid` (the
λ-derivative) and `ρ_v`."""
function interp_sheet_at_points(x_grid::AbstractArray{T,4}, ρv::AbstractArray{T,3},
                                pts::AbstractMatrix{T}, cl, res::Int; eps::Real=1e-7) where {T}
    backend = get_backend(x_grid); off = _offsets_on(x_grid)
    mv(x)=(y=similar(x_grid,eltype(x),size(x)); copyto!(y,x); y)
    ρg = KernelAbstractions.zeros(backend, Float64, size(pts,1))
    _interp_fwd!(backend)(ρg, x_grid, ρv, off, mv(pts), mv(cl.perm), mv(cl.cell_start),
        T(cl.o1),T(cl.o2),T(cl.o3),T(cl.h),cl.d1,cl.d2,cl.d3, res, T(eps); ndrange=(6,res-1,res-1,res-1))
    synchronize(backend)
    return ρg
end

function ChainRulesCore.rrule(::typeof(interp_sheet_at_points), x_grid::AbstractArray{T,4},
        ρv::AbstractArray{T,3}, pts::AbstractMatrix{T}, cl, res::Int; eps::Real=1e-7) where {T}
    ρg = interp_sheet_at_points(x_grid, ρv, pts, cl, res; eps)
    backend = get_backend(x_grid); off = _offsets_on(x_grid)
    mv(x)=(y=similar(x_grid,eltype(x),size(x)); copyto!(y,x); y)
    ptsb=mv(pts); permb=mv(cl.perm); cstartb=mv(cl.cell_start)
    function interp_pullback(Δ)
        ρ̄g = Δ isa ChainRulesCore.AbstractZero ? KernelAbstractions.zeros(backend,Float64,size(pts,1)) :
             (y=KernelAbstractions.zeros(backend,Float64,size(pts,1)); copyto!(y, Float64.(unthunk(Δ))); y)
        x̄ = KernelAbstractions.zeros(backend, Float64, res,res,res,3); ρ̄v = KernelAbstractions.zeros(backend, Float64, res,res,res)
        _interp_bwd!(backend)(x̄, ρ̄v, ρ̄g, x_grid, ρv, off, ptsb, permb, cstartb,
            T(cl.o1),T(cl.o2),T(cl.o3),T(cl.h),cl.d1,cl.d2,cl.d3, res, T(eps); ndrange=(6,res-1,res-1,res-1))
        synchronize(backend)
        return (NoTangent(), T.(x̄), T.(ρ̄v), NoTangent(), NoTangent(), NoTangent())
    end
    return ρg, interp_pullback
end
