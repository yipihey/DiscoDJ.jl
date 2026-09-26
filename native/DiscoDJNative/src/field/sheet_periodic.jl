# ── Periodic phase-space sheet: element geometry, exact stream counts, mesh density ──
#
# The kernels in sheet_density.jl / sheet_query.jl tessellate only the (res−1)³ interior
# cubes (non-periodic connectivity, built for lightcone embeddings).  For a periodic
# simulation box every Lagrangian cube is an element, including those that wrap around
# the box.  This file provides the periodic counterparts, with the same tessellation
# (`_TETS`/`_CUBE_CORNERS`, 6 Kuhn tetrahedra per cube) and the same per-tet
# determinant / Cramer barycentric test as `locate_points_in_sheet`.
#
# Input is the Lagrangian displacement field ψ::(n,n,n,3) (x = q + ψ, ψ periodic and
# *unwrapped*, i.e. continuous across the box faces).  The vertex of cube (i,j,k) with
# lattice offset (a,b,c) sits at  y = (i−1+a, j−1+b, k−1+c)·Δq + ψ[mod1(i+a), …]  — an
# unwrapped Eulerian position, so tetrahedra that straddle the box face stay compact.
# Query points / mesh nodes live in [0,L)³ and are matched to a tetrahedron through the
# periodic image that falls inside its bounding box.
#
# Units: ρ̄ = 1.  A cube carries mass Δq³, each tetrahedron Δq³/6, so a tetrahedron of
# Eulerian volume V_T has stream density Δq³/(6|V_T|) = Δq³/|det E|.
#
#   sheet_elements_periodic(ψ, L)            → (; V, nflip, centroid)  per element
#   PeriodicCellList(pts, L, h)              → cell list for points in [0,L)³
#   sheet_query_periodic(ψ, L, pts, cl)      → (; density, nstream)    at points
#   sheet_mesh_periodic(ψ, L, ng)            → (; density, nstream)    at mesh nodes (j−1)·L/ng
#   refine_displacement(ψ, level)            → band-limited ψ on a 2^level finer lattice
#   sample_trilinear_periodic(f, pts, L)     → node-centred mesh field at points
#
# Forward-only analysis tools (not on the AD tape); CPU and CUDA via KernelAbstractions.

export sheet_elements_periodic, PeriodicCellList, sheet_query_periodic, sheet_mesh_periodic, sheet_locate_mesh_periodic,
       refine_displacement, sample_trilinear_periodic, tet_orientation_signs

"""    tet_orientation_signs() -> NTuple{6,Int}

Sign of det of the Lagrangian edge matrix of each of the 6 tetrahedra (`_TETS`): the
signed volume of tetrahedron t on the undeformed lattice is `sign_t · Δq³/6`."""
function tet_orientation_signs()
    ntuple(6) do t
        M = [_TET_OFFSETS[t, v+1, d] - _TET_OFFSETS[t, 1, d] for d in 1:3, v in 1:3]
        dt = M[1,1]*(M[2,2]*M[3,3]-M[2,3]*M[3,2]) - M[1,2]*(M[2,1]*M[3,3]-M[2,3]*M[3,1]) +
             M[1,3]*(M[2,1]*M[3,2]-M[2,2]*M[3,1])
        Int(sign(dt))
    end
end

# unwrapped Eulerian vertex v of tet t of cube (i,j,k) (1-based)
@inline function _pvert(ψ, off, t, v, i, j, k, n, dq)
    a = off[t, v, 1]; b = off[t, v, 2]; c = off[t, v, 3]
    ii = mod1(i + a, n); jj = mod1(j + b, n); kk = mod1(k + c, n)
    return ((i - 1 + a) * dq + ψ[ii, jj, kk, 1],
            (j - 1 + b) * dq + ψ[ii, jj, kk, 2],
            (k - 1 + c) * dq + ψ[ii, jj, kk, 3])
end

# ── element geometry ──────────────────────────────────────────────────────────
@kernel function _elements_periodic!(V, nflip, cen, @Const(ψ), @Const(off), @Const(sgn), n::Int, dq, L)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Vs = zero(eltype(V)); nf = Int8(0)
        for t in 1:6
            y1 = _pvert(ψ, off, t, 1, i, j, k, n, dq); y2 = _pvert(ψ, off, t, 2, i, j, k, n, dq)
            y3 = _pvert(ψ, off, t, 3, i, j, k, n, dq); y4 = _pvert(ψ, off, t, 4, i, j, k, n, dq)
            e1x = y2[1]-y1[1]; e1y = y2[2]-y1[2]; e1z = y2[3]-y1[3]
            e2x = y3[1]-y1[1]; e2y = y3[2]-y1[2]; e2z = y3[3]-y1[3]
            e3x = y4[1]-y1[1]; e3y = y4[2]-y1[2]; e3z = y4[3]-y1[3]
            detf = e1x*(e2y*e3z-e2z*e3y) - e1y*(e2x*e3z-e2z*e3x) + e1z*(e2x*e3y-e2y*e3x)
            Vt = sgn[t] * detf / 6
            Vs += Vt
            Vt <= 0 && (nf += Int8(1))
        end
        V[i, j, k] = Vs; nflip[i, j, k] = nf
        # centroid of the 8 cube corners (unwrapped), then wrapped into [0,L)
        cx = zero(eltype(cen)); cy = cx; cz = cx
        for a in 0:1, b in 0:1, c in 0:1
            ii = mod1(i + a, n); jj = mod1(j + b, n); kk = mod1(k + c, n)
            cx += (i - 1 + a) * dq + ψ[ii, jj, kk, 1]
            cy += (j - 1 + b) * dq + ψ[ii, jj, kk, 2]
            cz += (k - 1 + c) * dq + ψ[ii, jj, kk, 3]
        end
        cen[i, j, k, 1] = mod(cx / 8, L); cen[i, j, k, 2] = mod(cy / 8, L); cen[i, j, k, 3] = mod(cz / 8, L)
    end
end

"""
    sheet_elements_periodic(ψ, L) -> (; V, nflip, centroid)

Per Lagrangian cube (element) of the periodic sheet: `V::(n,n,n)` the exact Eulerian
volume (sum of the 6 orientation-corrected tetrahedron volumes; `sum(V) == L³` for any
periodic ψ), `nflip::(n,n,n) Int8` the number of its tetrahedra with non-positive
oriented volume (0 ⇔ no tetrahedron of the element has been inverted), and
`centroid::(n,n,n,3)` the mean of its 8 corners wrapped into [0,L)³.  The element's
stream density is `(L/n)³ / V`.
"""
function sheet_elements_periodic(ψ::AbstractArray{T,4}, L::Real) where {T}
    n = size(ψ, 1); backend = get_backend(ψ)
    off = _offsets_on(ψ)
    sgn = similar(ψ, Int, 6); copyto!(sgn, collect(tet_orientation_signs()))
    V = KernelAbstractions.zeros(backend, Float64, n, n, n)
    nflip = KernelAbstractions.zeros(backend, Int8, n, n, n)
    cen = similar(ψ, T, n, n, n, 3)
    _elements_periodic!(backend)(V, nflip, cen, ψ, off, sgn, n, T(L / n), T(L); ndrange=(n, n, n))
    synchronize(backend)
    return (V=V, nflip=nflip, centroid=cen)
end

# ── periodic cell list on query points in [0,L)³ ───────────────────────────────
"""
    PeriodicCellList(pts::(N,3), L, h)

Counting-sort of points in [0,L)³ into `d³` periodic cells of size `L/d` (`d = floor(L/h)`),
with Int32 indices (memory: ~8 bytes per point).  Host-side; `perm[cstart[c]:cstart[c+1]-1]`
are the points in cell c."""
struct PeriodicCellList
    d::Int
    hc::Float64
    cstart::Vector{Int64}
    perm::Vector{Int32}
end

function PeriodicCellList(pts::AbstractMatrix, L::Real, h::Real)
    N = size(pts, 1)
    N < typemax(Int32) || error("PeriodicCellList: too many points for Int32 indices")
    d = max(1, floor(Int, L / h)); hc = L / d
    cid = Vector{Int32}(undef, N); counts = zeros(Int64, d^3)
    @inbounds for g in 1:N
        cx = mod(floor(Int, pts[g, 1] / hc), d); cy = mod(floor(Int, pts[g, 2] / hc), d); cz = mod(floor(Int, pts[g, 3] / hc), d)
        c = cx + d * (cy + d * cz) + 1; cid[g] = c; counts[c] += 1
    end
    cstart = Vector{Int64}(undef, d^3 + 1); cstart[1] = 1
    @inbounds for c in 1:d^3; cstart[c+1] = cstart[c] + counts[c]; end
    perm = Vector{Int32}(undef, N); fill_at = copy(cstart)
    @inbounds for g in 1:N; c = cid[g]; perm[fill_at[c]] = g; fill_at[c] += 1; end
    return PeriodicCellList(d, Float64(hc), cstart, perm)
end

# barycentric inclusion (closed tetrahedron, exact: no tolerance → a point on a shared face is
# counted by each tetrahedron containing it; this is a measure-zero event for generic positions)
@inline function _inside(dx, dy, dz, e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z, c23x, c23y, c23z, inv)
    l2 = (dx*c23x + dy*c23y + dz*c23z) * inv
    l3 = (e1x*(dy*e3z-dz*e3y) + e1y*(dz*e3x-dx*e3z) + e1z*(dx*e3y-dy*e3x)) * inv
    l4 = (e1x*(e2y*dz-e2z*dy) + e1y*(e2z*dx-e2x*dz) + e1z*(e2x*dy-e2y*dx)) * inv
    l1 = one(l2) - l2 - l3 - l4
    return (l1 >= 0) & (l2 >= 0) & (l3 >= 0) & (l4 >= 0)
end

@kernel function _query_periodic!(dens, nstr, @Const(ψ), @Const(off), @Const(pts), @Const(perm),
                                  @Const(cstart), n::Int, dq, L, d::Int, hc, dq3)
    t, i, j, k = @index(Global, NTuple)
    @inbounds begin
        y1 = _pvert(ψ, off, t, 1, i, j, k, n, dq); y2 = _pvert(ψ, off, t, 2, i, j, k, n, dq)
        y3 = _pvert(ψ, off, t, 3, i, j, k, n, dq); y4 = _pvert(ψ, off, t, 4, i, j, k, n, dq)
        e1x = y2[1]-y1[1]; e1y = y2[2]-y1[2]; e1z = y2[3]-y1[3]
        e2x = y3[1]-y1[1]; e2y = y3[2]-y1[2]; e2z = y3[3]-y1[3]
        e3x = y4[1]-y1[1]; e3y = y4[2]-y1[2]; e3z = y4[3]-y1[3]
        c23x = e2y*e3z-e2z*e3y; c23y = e2z*e3x-e2x*e3z; c23z = e2x*e3y-e2y*e3x
        detf = e1x*c23x + e1y*c23y + e1z*c23z
        if detf != 0
            inv = one(detf) / detf; ρT = dq3 / abs(detf)
            ca = floor(Int, min(y1[1], y2[1], y3[1], y4[1]) / hc); cb = floor(Int, max(y1[1], y2[1], y3[1], y4[1]) / hc)
            da = floor(Int, min(y1[2], y2[2], y3[2], y4[2]) / hc); db = floor(Int, max(y1[2], y2[2], y3[2], y4[2]) / hc)
            ea = floor(Int, min(y1[3], y2[3], y3[3], y4[3]) / hc); eb = floor(Int, max(y1[3], y2[3], y3[3], y4[3]) / hc)
            for cz in ea:eb, cy in da:db, cx in ca:cb
                # unwrapped cell (cx,cy,cz) = wrapped cell + d·(image shift)
                wx = mod(cx, d); wy = mod(cy, d); wz = mod(cz, d)
                sx = (cx - wx) ÷ d; sy = (cy - wy) ÷ d; sz = (cz - wz) ÷ d
                c = wx + d * (wy + d * wz) + 1
                for idx in cstart[c]:(cstart[c+1]-1)
                    g = perm[idx]
                    dx = pts[g, 1] + sx * L - y1[1]; dy = pts[g, 2] + sy * L - y1[2]; dz = pts[g, 3] + sz * L - y1[3]
                    if _inside(dx, dy, dz, e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z, c23x, c23y, c23z, inv)
                        KernelAbstractions.@atomic dens[g] += Float64(ρT)
                        KernelAbstractions.@atomic nstr[g] += Int32(1)
                    end
                end
            end
        end
    end
end

"""
    sheet_query_periodic(ψ, L, pts, cl) -> (; density, nstream)

Exact AHK sheet density `Σ_{T∋x} Δq³/|det E_T|` (all streams, ρ̄ = 1) and stream multiplicity
(number of tetrahedra containing the point) at points `pts::(N,3)` in [0,L)³, for the
periodic sheet with displacement ψ.  `cl = PeriodicCellList(pts, L, h)` (h ≈ L/n is a good
choice).  In a periodic box every point is covered: `nstream ≥ 1`, odd for generic points."""
# move `x` to ψ's backend/eltype only when needed (no copy for host arrays already in place)
_like(ψ, x::AbstractArray, ::Type{E}) where {E} =
    (typeof(x) <: Array && ψ isa Array && eltype(x) === E) ? x :
    (y = similar(ψ, E, size(x)); copyto!(y, E.(x)); y)

function sheet_query_periodic(ψ::AbstractArray{T,4}, L::Real, pts::AbstractMatrix, cl::PeriodicCellList) where {T}
    n = size(ψ, 1); backend = get_backend(ψ); off = _offsets_on(ψ)
    N = size(pts, 1)
    dens = KernelAbstractions.zeros(backend, Float64, N)
    nstr = KernelAbstractions.zeros(backend, Int32, N)
    dq = T(L / n)
    _query_periodic!(backend)(dens, nstr, ψ, off, _like(ψ, pts, T), _like(ψ, cl.perm, Int32),
                              _like(ψ, cl.cstart, Int64), n, dq, T(L),
                              cl.d, T(cl.hc), dq^3; ndrange=(6, n, n, n))
    synchronize(backend)
    return (density=dens, nstream=nstr)
end

# ── mesh-node density / stream count (the mesh is its own cell list) ─────────
@kernel function _mesh_periodic!(dens, nstr, @Const(ψ), @Const(off), n::Int, dq, ng::Int, h, dq3)
    t, i, j, k = @index(Global, NTuple)
    @inbounds begin
        y1 = _pvert(ψ, off, t, 1, i, j, k, n, dq); y2 = _pvert(ψ, off, t, 2, i, j, k, n, dq)
        y3 = _pvert(ψ, off, t, 3, i, j, k, n, dq); y4 = _pvert(ψ, off, t, 4, i, j, k, n, dq)
        e1x = y2[1]-y1[1]; e1y = y2[2]-y1[2]; e1z = y2[3]-y1[3]
        e2x = y3[1]-y1[1]; e2y = y3[2]-y1[2]; e2z = y3[3]-y1[3]
        e3x = y4[1]-y1[1]; e3y = y4[2]-y1[2]; e3z = y4[3]-y1[3]
        c23x = e2y*e3z-e2z*e3y; c23y = e2z*e3x-e2x*e3z; c23z = e2x*e3y-e2y*e3x
        detf = e1x*c23x + e1y*c23y + e1z*c23z
        if detf != 0
            inv = one(detf) / detf; ρT = dq3 / abs(detf)
            xa = ceil(Int, min(y1[1], y2[1], y3[1], y4[1]) / h); xb = floor(Int, max(y1[1], y2[1], y3[1], y4[1]) / h)
            ya = ceil(Int, min(y1[2], y2[2], y3[2], y4[2]) / h); yb = floor(Int, max(y1[2], y2[2], y3[2], y4[2]) / h)
            za = ceil(Int, min(y1[3], y2[3], y3[3], y4[3]) / h); zb = floor(Int, max(y1[3], y2[3], y3[3], y4[3]) / h)
            for z in za:zb, y in ya:yb, x in xa:xb
                dx = x * h - y1[1]; dy = y * h - y1[2]; dz = z * h - y1[3]
                if _inside(dx, dy, dz, e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z, c23x, c23y, c23z, inv)
                    I = mod(x, ng) + 1; J = mod(y, ng) + 1; K = mod(z, ng) + 1
                    KernelAbstractions.@atomic dens[I, J, K] += Float64(ρT)
                    KernelAbstractions.@atomic nstr[I, J, K] += Int32(1)
                end
            end
        end
    end
end

"""
    sheet_mesh_periodic(ψ, L, ng) -> (; density, nstream)

Point-sampled AHK sheet density (all streams, ρ̄ = 1, exact tetrahedron volumes) and stream
multiplicity at the `ng³` mesh nodes `x = (I−1, J−1, K−1)·L/ng` of the periodic box.  No
deposit window: the values are the exact piecewise-constant sheet density at the nodes."""
function sheet_mesh_periodic(ψ::AbstractArray{T,4}, L::Real, ng::Int) where {T}
    n = size(ψ, 1); backend = get_backend(ψ); off = _offsets_on(ψ)
    dens = KernelAbstractions.zeros(backend, Float64, ng, ng, ng)
    nstr = KernelAbstractions.zeros(backend, Int32, ng, ng, ng)
    dq = T(L / n)
    _mesh_periodic!(backend)(dens, nstr, ψ, off, n, dq, ng, T(L / ng), dq^3; ndrange=(6, n, n, n))
    synchronize(backend)
    return (density=dens, nstream=nstr)
end

# ── element identity at mesh nodes (Eulerian "label" transport) ───────────────
@kernel function _locate_mesh_periodic!(elem, nstr, @Const(ψ), @Const(off), n::Int, dq, ng::Int, h)
    t, i, j, k = @index(Global, NTuple)
    @inbounds begin
        y1 = _pvert(ψ, off, t, 1, i, j, k, n, dq); y2 = _pvert(ψ, off, t, 2, i, j, k, n, dq)
        y3 = _pvert(ψ, off, t, 3, i, j, k, n, dq); y4 = _pvert(ψ, off, t, 4, i, j, k, n, dq)
        e1x = y2[1]-y1[1]; e1y = y2[2]-y1[2]; e1z = y2[3]-y1[3]
        e2x = y3[1]-y1[1]; e2y = y3[2]-y1[2]; e2z = y3[3]-y1[3]
        e3x = y4[1]-y1[1]; e3y = y4[2]-y1[2]; e3z = y4[3]-y1[3]
        c23x = e2y*e3z-e2z*e3y; c23y = e2z*e3x-e2x*e3z; c23z = e2x*e3y-e2y*e3x
        detf = e1x*c23x + e1y*c23y + e1z*c23z
        if detf != 0
            inv = one(detf) / detf
            eid = Int32(i + n * (j - 1 + n * (k - 1)))           # column-major element index
            xa = ceil(Int, min(y1[1], y2[1], y3[1], y4[1]) / h); xb = floor(Int, max(y1[1], y2[1], y3[1], y4[1]) / h)
            ya = ceil(Int, min(y1[2], y2[2], y3[2], y4[2]) / h); yb = floor(Int, max(y1[2], y2[2], y3[2], y4[2]) / h)
            za = ceil(Int, min(y1[3], y2[3], y3[3], y4[3]) / h); zb = floor(Int, max(y1[3], y2[3], y3[3], y4[3]) / h)
            for z in za:zb, y in ya:yb, x in xa:xb
                dx = x * h - y1[1]; dy = y * h - y1[2]; dz = z * h - y1[3]
                if _inside(dx, dy, dz, e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z, c23x, c23y, c23z, inv)
                    I = mod(x, ng) + 1; J = mod(y, ng) + 1; K = mod(z, ng) + 1
                    KernelAbstractions.@atomic nstr[I, J, K] += Int32(1)
                    elem[I, J, K] = eid                          # meaningful only where nstr == 1
                end
            end
        end
    end
end

"""
    sheet_locate_mesh_periodic(ψ, L, ng) -> (; element, nstream)

For every mesh node `(I−1, J−1, K−1)·L/ng`: the stream multiplicity and, where it is 1
(single-stream), the column-major linear index of the Lagrangian element (cube) whose
tetrahedron contains the node.  Where `nstream > 1` the element is an arbitrary one of them."""
function sheet_locate_mesh_periodic(ψ::AbstractArray{T,4}, L::Real, ng::Int) where {T}
    n = size(ψ, 1); backend = get_backend(ψ); off = _offsets_on(ψ)
    elem = KernelAbstractions.zeros(backend, Int32, ng, ng, ng)
    nstr = KernelAbstractions.zeros(backend, Int32, ng, ng, ng)
    _locate_mesh_periodic!(backend)(elem, nstr, ψ, off, n, T(L / n), ng, T(L / ng); ndrange=(6, n, n, n))
    synchronize(backend)
    return (element=elem, nstream=nstr)
end

# ── band-limited refinement of ψ (Fourier zero-padding) ───────────────────────
"""
    refine_displacement(ψ, level) -> ψ on a (n·2^level)³ lattice

Band-limited (Fourier) interpolation of each displacement component onto a 2^level finer
Lagrangian lattice (the Nyquist planes of the coarse grid are dropped).  Exact for fields
that are band-limited on the coarse grid, such as nLPT displacements; for N-body
displacements it defines the refined sheet by the same interpolation.  Host (FFTW)."""
function refine_displacement(ψ::AbstractArray{T,4}, level::Int) where {T}
    level == 0 && return ψ
    n = size(ψ, 1); e = n * 2^level; h = n ÷ 2
    out = Array{T,4}(undef, e, e, e, 3)
    lo = 1:h; hs = h+2:n; hd = e-h+2:e
    for c in 1:3
        F = FFTW.rfft(Array(ψ[:, :, :, c]))          # (h+1, n, n): half axis first (FFTW.jl)
        G = zeros(Complex{T}, e ÷ 2 + 1, e, e)
        for (ys, yd) in ((lo, lo), (hs, hd)), (zs, zd) in ((lo, lo), (hs, hd))
            G[1:h, yd, zd] .= F[1:h, ys, zs]
        end
        G .*= T((e / n)^3)
        out[:, :, :, c] .= FFTW.irfft(G, e)
    end
    return out
end

# ── trilinear sampling of a node-centred periodic mesh field ──────────────────
"""    sample_trilinear_periodic(f::(ng,ng,ng), pts::(N,3), L) -> (N,)"""
function sample_trilinear_periodic(f::AbstractArray{<:Real,3}, pts::AbstractMatrix, L::Real)
    ng = size(f, 1); s = ng / L; N = size(pts, 1)
    out = Vector{Float64}(undef, N)
    Threads.@threads for g in 1:N
        @inbounds begin
            fx = pts[g, 1] * s; fy = pts[g, 2] * s; fz = pts[g, 3] * s
            ix = floor(Int, fx); iy = floor(Int, fy); iz = floor(Int, fz)
            dx = fx - ix; dy = fy - iy; dz = fz - iz
            i0 = mod(ix, ng) + 1; j0 = mod(iy, ng) + 1; k0 = mod(iz, ng) + 1
            i1 = mod(ix + 1, ng) + 1; j1 = mod(iy + 1, ng) + 1; k1 = mod(iz + 1, ng) + 1
            out[g] = f[i0,j0,k0]*(1-dx)*(1-dy)*(1-dz) + f[i1,j0,k0]*dx*(1-dy)*(1-dz) +
                     f[i0,j1,k0]*(1-dx)*dy*(1-dz) + f[i0,j0,k1]*(1-dx)*(1-dy)*dz +
                     f[i1,j1,k0]*dx*dy*(1-dz) + f[i1,j0,k1]*dx*(1-dy)*dz +
                     f[i0,j1,k1]*(1-dx)*dy*dz + f[i1,j1,k1]*dx*dy*dz
        end
    end
    return out
end
