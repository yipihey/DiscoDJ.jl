# ── Stream-resolved velocity moments of the periodic phase-space sheet at mesh nodes ──
#
# At a point x the sheet represents the distribution function exactly as a sum over streams,
#   f(x, v) = Σ_s ρ_s δ_D(v − v_s(x)),
# with ρ_s = Δq³/|det E_T| the density of the tetrahedron T of stream s that contains x and v_s(x)
# the barycentric (P1) interpolant of the vertex velocities in T.  Its first two moments are
#   ρ = Σ ρ_s,   v̄ = Σ ρ_s v_s / ρ,   σ_ij = Σ ρ_s (v_s − v̄)_i (v_s − v̄)_j / ρ,
# so σ ≡ 0 wherever a single stream covers the node (no cell-averaging "dispersion" from velocity
# gradients, unlike any particle deposit).  σ is accumulated about v̄ in a second pass: just after
# shell crossing σ ≪ |v̄|², and Σρvv − ρv̄v̄ would cancel catastrophically in Float32.
#
# Same tessellation, vertex convention and closed-tetrahedron inclusion test as
# `sheet_mesh_periodic` (sheet_periodic.jl): ψ is the unwrapped displacement on the (n,n,n)
# Lagrangian lattice, nodes are x = (I−1, J−1, K−1)·L/ng.  An optional second vertex field
# `acc` (e.g. the kinematic acceleration d²x/dt²) is averaged like v̄ in the first pass.

export sheet_moments_mesh_periodic

@inline function _pval(v, off, t, vtx, i, j, k, n)
    ii = mod1(i + off[t, vtx, 1], n); jj = mod1(j + off[t, vtx, 2], n); kk = mod1(k + off[t, vtx, 3], n)
    return (v[ii, jj, kk, 1], v[ii, jj, kk, 2], v[ii, jj, kk, 3])
end

# barycentric coordinates (l1..l4) of the offset d = x − y1 (same algebra as `_inside`)
@inline function _bary4(dx, dy, dz, e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z, c23x, c23y, c23z, inv)
    l2 = (dx*c23x + dy*c23y + dz*c23z) * inv
    l3 = (e1x*(dy*e3z-dz*e3y) + e1y*(dz*e3x-dx*e3z) + e1z*(dx*e3y-dy*e3x)) * inv
    l4 = (e1x*(e2y*dz-e2z*dy) + e1y*(e2z*dx-e2x*dz) + e1z*(e2x*dy-e2y*dx)) * inv
    return (one(l2) - l2 - l3 - l4, l2, l3, l4)
end

@inline _interp(l, a, b, c, d, comp) = l[1] * a[comp] + l[2] * b[comp] + l[3] * c[comp] + l[4] * d[comp]

# pass = 1: ρ, n_stream, Σρv (→ m1), Σρa (→ ma, if HASACC);  pass = 2: Σρ(v−v̄)(v−v̄) (→ S, 6 comps)
@kernel function _moments_periodic!(dens, nstr, m1, ma, S, @Const(vbar), @Const(ψ), @Const(vel), @Const(acc),
                                    @Const(off), n::Int, dq, ng::Int, h, dq3, ::Val{PASS}, ::Val{HASACC}) where {PASS, HASACC}
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
            v1 = _pval(vel, off, t, 1, i, j, k, n); v2 = _pval(vel, off, t, 2, i, j, k, n)
            v3 = _pval(vel, off, t, 3, i, j, k, n); v4 = _pval(vel, off, t, 4, i, j, k, n)
            xa = ceil(Int, min(y1[1], y2[1], y3[1], y4[1]) / h); xb = floor(Int, max(y1[1], y2[1], y3[1], y4[1]) / h)
            ya = ceil(Int, min(y1[2], y2[2], y3[2], y4[2]) / h); yb = floor(Int, max(y1[2], y2[2], y3[2], y4[2]) / h)
            za = ceil(Int, min(y1[3], y2[3], y3[3], y4[3]) / h); zb = floor(Int, max(y1[3], y2[3], y3[3], y4[3]) / h)
            for z in za:zb, y in ya:yb, x in xa:xb
                dx = x * h - y1[1]; dy = y * h - y1[2]; dz = z * h - y1[3]
                if _inside(dx, dy, dz, e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z, c23x, c23y, c23z, inv)
                    I = mod(x, ng) + 1; J = mod(y, ng) + 1; K = mod(z, ng) + 1
                    l = _bary4(dx, dy, dz, e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z, c23x, c23y, c23z, inv)
                    vx = _interp(l, v1, v2, v3, v4, 1); vy = _interp(l, v1, v2, v3, v4, 2); vz = _interp(l, v1, v2, v3, v4, 3)
                    if PASS == 1
                        TA = eltype(dens); r = TA(ρT)
                        KernelAbstractions.@atomic dens[I, J, K] += r
                        KernelAbstractions.@atomic nstr[I, J, K] += Int32(1)
                        KernelAbstractions.@atomic m1[I, J, K, 1] += r * TA(vx)
                        KernelAbstractions.@atomic m1[I, J, K, 2] += r * TA(vy)
                        KernelAbstractions.@atomic m1[I, J, K, 3] += r * TA(vz)
                        if HASACC
                            a1 = _pval(acc, off, t, 1, i, j, k, n); a2 = _pval(acc, off, t, 2, i, j, k, n)
                            a3 = _pval(acc, off, t, 3, i, j, k, n); a4 = _pval(acc, off, t, 4, i, j, k, n)
                            KernelAbstractions.@atomic ma[I, J, K, 1] += r * TA(_interp(l, a1, a2, a3, a4, 1))
                            KernelAbstractions.@atomic ma[I, J, K, 2] += r * TA(_interp(l, a1, a2, a3, a4, 2))
                            KernelAbstractions.@atomic ma[I, J, K, 3] += r * TA(_interp(l, a1, a2, a3, a4, 3))
                        end
                    elseif nstr[I, J, K] > 1                     # single stream: σ = 0 exactly (no round-off)
                        TA = eltype(S); r = TA(ρT)
                        ux = TA(vx) - vbar[I, J, K, 1]; uy = TA(vy) - vbar[I, J, K, 2]; uz = TA(vz) - vbar[I, J, K, 3]
                        KernelAbstractions.@atomic S[I, J, K, 1] += r * ux * ux
                        KernelAbstractions.@atomic S[I, J, K, 2] += r * uy * uy
                        KernelAbstractions.@atomic S[I, J, K, 3] += r * uz * uz
                        KernelAbstractions.@atomic S[I, J, K, 4] += r * ux * uy
                        KernelAbstractions.@atomic S[I, J, K, 5] += r * ux * uz
                        KernelAbstractions.@atomic S[I, J, K, 6] += r * uy * uz
                    end
                end
            end
        end
    end
end

"""
    sheet_moments_mesh_periodic(ψ, vel, L, ng; acc=nothing, Tacc=Float64)
        -> (; density, nstream, vbar, sigma, abar)

Point-sampled first and second velocity moments of the periodic AHK sheet at the `ng³` mesh
nodes `(I−1, J−1, K−1)·L/ng`: `density` (ρ̄ = 1, all streams; identical to
`sheet_mesh_periodic`), `nstream`, the mass-weighted mean velocity `vbar::(ng,ng,ng,3)` and the
velocity dispersion tensor `sigma::(ng,ng,ng,6)` in the order (xx, yy, zz, xy, xz, yz), per unit
mass (ρσ is the kinetic stress about the mean flow).  `vel` (and the optional `acc`, returned as
the mass-weighted mean `abar`, else `nothing`) are (n,n,n,3) vertex fields on the Lagrangian
lattice of ψ, interpolated linearly inside each tetrahedron.  `sigma` is exactly zero where
`nstream == 1`.  Accumulators and outputs are `Tacc` (Float32 halves the memory: 13·ng³
values with `acc`, 10·ng³ without); σ is accumulated about v̄ in a second pass."""
function sheet_moments_mesh_periodic(ψ::AbstractArray{T,4}, vel::AbstractArray{T,4}, L::Real, ng::Int;
                                     acc=nothing, Tacc::Type{<:AbstractFloat}=Float64) where {T}
    n = size(ψ, 1); backend = get_backend(ψ); off = _offsets_on(ψ)
    size(vel) == size(ψ) || error("vel must have the shape of ψ")
    hasacc = acc !== nothing
    hasacc && size(acc) != size(ψ) && error("acc must have the shape of ψ")
    dens = KernelAbstractions.zeros(backend, Tacc, ng, ng, ng)
    nstr = KernelAbstractions.zeros(backend, Int32, ng, ng, ng)
    m1 = KernelAbstractions.zeros(backend, Tacc, ng, ng, ng, 3)
    ma = hasacc ? KernelAbstractions.zeros(backend, Tacc, ng, ng, ng, 3) : m1
    dq = T(L / n); h = T(L / ng)
    kern = _moments_periodic!(backend)
    kern(dens, nstr, m1, ma, m1, m1, ψ, vel, hasacc ? acc : vel, off, n, dq, ng, h, dq^3, Val(1), Val(hasacc);
         ndrange=(6, n, n, n))
    synchronize(backend)
    # every node of a periodic box is covered by ≥ 1 stream, so ρ > 0; guard anyway
    rinv = map(r -> r > 0 ? inv(r) : zero(r), dens)
    m1 .*= reshape(rinv, ng, ng, ng, 1)                      # → v̄
    hasacc && (ma .*= reshape(rinv, ng, ng, ng, 1))          # → ā
    S = KernelAbstractions.zeros(backend, Tacc, ng, ng, ng, 6)
    kern(dens, nstr, m1, ma, S, m1, ψ, vel, hasacc ? acc : vel, off, n, dq, ng, h, dq^3, Val(2), Val(hasacc);
         ndrange=(6, n, n, n))
    synchronize(backend)
    S .*= reshape(rinv, ng, ng, ng, 1)                       # → σ (per unit mass)
    return (density=dens, nstream=nstr, vbar=m1, sigma=S, abar=hasacc ? ma : nothing)
end

# ── Conservative tracer deposits of the P1 sheet, with exact divergences ──────────────────────
# Tracers at fixed barycentric positions λ inside every tetrahedron carry equal shares m of its mass
# and move with the P1 sheet kinematics exactly (x = Σλ y_v, dx/dt = Σλ v_v, d²x/dt² = Σλ a_v).
# With a C² deposit kernel W (PCS, the cubic B-spline) the node fields
#     ρ_g = Σ m W(x_g − x),   flux_g = Σ m v·∇W,   stress_g = Σ m v_i v_j ∂_i∂_j W,   acc_g = Σ m a·∇W
# satisfy  ∂_t ρ_g = −flux_g  and  ∂²_t ρ_g = stress_g − acc_g  IDENTICALLY at every node, so any linear
# operation applied afterwards (e.g. a top-hat by FFT) preserves the mass budget exactly.  Spectral
# derivatives of deposited fields do not: the tracers' sub-cell structure aliases to low k.
# Nodes at (I−1)·L/ng; ρ̄ = 1 units (Σ_g ρ_g = ng³).

export sheet_tracer_deposit_periodic, sheet_tracer_bary, TRACER_MODES

"""Barycentric tracer positions of level `s`: λ = (i, j, k, l .+ 1/4)/(s + 1) for i+j+k+l = s,
`binomial(s+3, 3)` points (s = 0: centroid), all strictly inside the tetrahedron."""
function sheet_tracer_bary(s::Int)
    pts = NTuple{4,Float64}[]
    for i in 0:s, j in 0:s-i, k in 0:s-i-j
        l = s - i - j - k
        push!(pts, ((i + 0.25) / (s + 1), (j + 0.25) / (s + 1), (k + 0.25) / (s + 1), (l + 0.25) / (s + 1)))
    end
    return [p[c] for p in pts, c in 1:4]
end

"""Deposit modes of `sheet_tracer_deposit_periodic` (node value Σ over tracers of m × …):
  :rho     W                         :flux    v·∇W              :stress  v_i v_j ∂_i∂_j W
  :acc     a·∇W   (needs acc)        :gdiv    g(x)·∇W   (needs G = g, (ng,ng,ng,3), PCS-gathered)
  :bulk    ū_i ū_j ∂_i∂_j W  (G = ū, (ng,ng,ng,3))      :disp    σ_ij(x) ∂_i∂_j W  (G = σ, (ng,ng,ng,6))
  :vx/:vy/:vz  v_c W  (momentum components)"""
const TRACER_MODES = (:rho, :flux, :stress, :acc, :gdiv, :bulk, :disp, :vx, :vy, :vz)

# cubic B-spline (PCS) and its first two derivatives, s = (x_node − x)/h
@inline _pcs0(s) = (a = abs(s); a < 1 ? (4 - 6a^2 + 3a^3) / 6 : (a < 2 ? (2 - a)^3 / 6 : zero(s)))
@inline _pcs1(s) = (a = abs(s); a < 1 ? (-12s + 9s * a) / 6 : (a < 2 ? -sign(s) * (2 - a)^2 / 2 : zero(s)))
@inline _pcs2(s) = (a = abs(s); a < 1 ? -2 + 3a : (a < 2 ? 2 - a : zero(s)))

# per-axis PCS weights of the 4 nodes i0−1 … i0+2 around grid coordinate g (i0 = floor(g)), and their
# (periodic, 1-based) node indices — computed once per tracer and shared by gathers and the deposit
@inline _pcsw(f, i0, g) = (f(i0 - 1 - g), f(i0 - g), f(i0 + 1 - g), f(i0 + 2 - g))
@inline _pcsi(i0, ng) = (mod(i0 - 1, ng) + 1, mod(i0, ng) + 1, mod(i0 + 1, ng) + 1, mod(i0 + 2, ng) + 1)

# PCS gather of components 1..3 / 1..6 of the mesh field G with precomputed weights / indices
# (explicit scalar accumulators: a tuple rebuilt through a closure capturing itself gets boxed)
@inline function _pcs_gather3(G, Ix, Iy, Iz, wx, wy, wz)
    TG = eltype(G); s1 = zero(TG); s2 = zero(TG); s3 = zero(TG)
    for oz in 1:4, oy in 1:4
        wyz = TG(wy[oy]) * TG(wz[oz]); J = Iy[oy]; K = Iz[oz]
        for ox in 1:4
            w = TG(wx[ox]) * wyz; I = Ix[ox]
            s1 += w * G[I, J, K, 1]; s2 += w * G[I, J, K, 2]; s3 += w * G[I, J, K, 3]
        end
    end
    return (s1, s2, s3)
end
@inline function _pcs_gather6(G, Ix, Iy, Iz, wx, wy, wz)
    TG = eltype(G); s1 = zero(TG); s2 = zero(TG); s3 = zero(TG); s4 = zero(TG); s5 = zero(TG); s6 = zero(TG)
    for oz in 1:4, oy in 1:4
        wyz = TG(wy[oy]) * TG(wz[oz]); J = Iy[oy]; K = Iz[oz]
        for ox in 1:4
            w = TG(wx[ox]) * wyz; I = Ix[ox]
            s1 += w * G[I, J, K, 1]; s2 += w * G[I, J, K, 2]; s3 += w * G[I, J, K, 3]
            s4 += w * G[I, J, K, 4]; s5 += w * G[I, J, K, 5]; s6 += w * G[I, J, K, 6]
        end
    end
    return (s1, s2, s3, s4, s5, s6)
end

# one tetrahedron (t, i, j, k): its tracers' deposit; ATOMIC = false for a thread-private mesh
@inline function _tracer_tet!(mesh, ψ, vel, acc, G, off, bary, nbary, n, dq, ng, invh, mt, mode, t, i, j, k,
                              ::Val{ATOMIC}) where {ATOMIC}
    @inbounds begin
        TA = eltype(mesh); T = typeof(dq)
        y1 = _pvert(ψ, off, t, 1, i, j, k, n, dq); y2 = _pvert(ψ, off, t, 2, i, j, k, n, dq)
        y3 = _pvert(ψ, off, t, 3, i, j, k, n, dq); y4 = _pvert(ψ, off, t, 4, i, j, k, n, dq)
        v1 = _pval(vel, off, t, 1, i, j, k, n); v2 = _pval(vel, off, t, 2, i, j, k, n)
        v3 = _pval(vel, off, t, 3, i, j, k, n); v4 = _pval(vel, off, t, 4, i, j, k, n)
        a1 = _pval(acc, off, t, 1, i, j, k, n); a2 = _pval(acc, off, t, 2, i, j, k, n)
        a3 = _pval(acc, off, t, 3, i, j, k, n); a4 = _pval(acc, off, t, 4, i, j, k, n)
        for b in 1:nbary
            l = (bary[b, 1], bary[b, 2], bary[b, 3], bary[b, 4])
            gx = _interp(l, y1, y2, y3, y4, 1) * invh; gy = _interp(l, y1, y2, y3, y4, 2) * invh
            gz = _interp(l, y1, y2, y3, y4, 3) * invh
            vx = _interp(l, v1, v2, v3, v4, 1); vy = _interp(l, v1, v2, v3, v4, 2); vz = _interp(l, v1, v2, v3, v4, 3)
            ix = floor(Int, gx); iy = floor(Int, gy); iz = floor(Int, gz)
            Ix = _pcsi(ix, ng); Iy = _pcsi(iy, ng); Iz = _pcsi(iz, ng)
            w0x = _pcsw(_pcs0, ix, gx); w0y = _pcsw(_pcs0, iy, gy); w0z = _pcsw(_pcs0, iz, gz)
            # node weight = c W + u·∇W + S:∂∂W
            c = zero(T); ux = zero(T); uy = zero(T); uz = zero(T)
            sxx = zero(T); syy = zero(T); szz = zero(T); sxy = zero(T); sxz = zero(T); syz = zero(T)
            if mode == 1
                c = one(T)
            elseif mode == 2
                ux = vx; uy = vy; uz = vz
            elseif mode == 3
                sxx = vx * vx; syy = vy * vy; szz = vz * vz; sxy = vx * vy; sxz = vx * vz; syz = vy * vz
            elseif mode == 4
                ux = _interp(l, a1, a2, a3, a4, 1); uy = _interp(l, a1, a2, a3, a4, 2); uz = _interp(l, a1, a2, a3, a4, 3)
            elseif mode == 5
                gg = _pcs_gather3(G, Ix, Iy, Iz, w0x, w0y, w0z)
                ux = T(gg[1]); uy = T(gg[2]); uz = T(gg[3])
            elseif mode == 6
                gg = _pcs_gather3(G, Ix, Iy, Iz, w0x, w0y, w0z)
                bx = T(gg[1]); by = T(gg[2]); bz = T(gg[3])
                sxx = bx * bx; syy = by * by; szz = bz * bz; sxy = bx * by; sxz = bx * bz; syz = by * bz
            elseif mode == 7
                gg = _pcs_gather6(G, Ix, Iy, Iz, w0x, w0y, w0z)
                sxx = T(gg[1]); syy = T(gg[2]); szz = T(gg[3]); sxy = T(gg[4]); sxz = T(gg[5]); syz = T(gg[6])
            else
                c = mode == 8 ? vx : (mode == 9 ? vy : vz)
            end
            ux *= invh; uy *= invh; uz *= invh
            ih2 = invh * invh
            sxx *= ih2; syy *= ih2; szz *= ih2; sxy *= 2ih2; sxz *= 2ih2; syz *= 2ih2
            w1x = _pcsw(_pcs1, ix, gx); w1y = _pcsw(_pcs1, iy, gy); w1z = _pcsw(_pcs1, iz, gz)
            w2x = _pcsw(_pcs2, ix, gx); w2y = _pcsw(_pcs2, iy, gy); w2z = _pcsw(_pcs2, iz, gz)
            for oz in 1:4
                for oy in 1:4
                    # weights factorised per (y, z) line: node weight = Σ_terms wx(term) · A(term)
                    a0 = w0y[oy] * w0z[oz]; ay1 = w1y[oy] * w0z[oz]; az1 = w0y[oy] * w1z[oz]
                    ayy = w2y[oy] * w0z[oz]; azz = w0y[oy] * w2z[oz]; ayz = w1y[oy] * w1z[oz]
                    k0 = c * a0 + uy * ay1 + uz * az1 + syy * ayy + szz * azz + syz * ayz   # × w0x
                    k1 = ux * a0 + sxy * ay1 + sxz * az1                                    # × w1x
                    k2 = sxx * a0                                                           # × w2x
                    for ox in 1:4
                        w = w0x[ox] * k0 + w1x[ox] * k1 + w2x[ox] * k2
                        if ATOMIC
                            KernelAbstractions.@atomic mesh[Ix[ox], Iy[oy], Iz[oz]] += TA(mt * w)
                        else
                            mesh[Ix[ox], Iy[oy], Iz[oz]] += TA(mt * w)
                        end
                    end
                end
            end
        end
    end
    return nothing
end

@kernel function _tracer_deposit!(mesh, @Const(ψ), @Const(vel), @Const(acc), @Const(G), @Const(off), @Const(bary),
                                  nbary::Int, n::Int, dq, ng::Int, invh, mt, mode::Int)
    t, i, j, k = @index(Global, NTuple)
    _tracer_tet!(mesh, ψ, vel, acc, G, off, bary, nbary, n, dq, ng, invh, mt, mode, t, i, j, k, Val(true))
end

# ── CPU path: thread-private meshes, plain adds, threaded reduction ──────────────────────────
# Float CAS atomics on a shared 512³ mesh (first-touched on one NUMA node) leave most cores stalled;
# instead each task deposits a Lagrangian k-slab into one of `nbuf` private meshes taken from a pool.
const _DEPOSIT_BUFFERS = Dict{Tuple{DataType,Int,Int},Vector{Array}}()
_deposit_nbuf(ng, TA) = clamp(min(Threads.nthreads(),
    floor(Int, parse(Float64, get(ENV, "DISCODJ_DEPOSIT_BUFFER_GB", "64")) * 2^30 / (sizeof(TA) * ng^3))), 1, 256)

# function barrier: concrete argument types for the per-tetrahedron loop of one Lagrangian k-slab
function _tracer_slab!(B::Array{TA,3}, ψ, vel, acc, G, off, bary, nbary::Int, n::Int, dq::T, ng::Int, invh::T, mt::T,
                       mode::Int, k::Int) where {TA, T}
    @inbounds for j in 1:n, i in 1:n, t in 1:6
        _tracer_tet!(B, ψ, vel, acc, G, off, bary, nbary, n, dq, ng, invh, mt, mode, t, i, j, k, Val(false))
    end
    return nothing
end

function _tracer_deposit_cpu!(mesh::Array{TA,3}, ψ, vel, acc, G, off, bary, nbary, n, dq, ng, invh, mt, mode) where {TA}
    nb = _deposit_nbuf(ng, TA)
    bufs = get!(() -> Array[], _DEPOSIT_BUFFERS, (TA, ng, nb))
    while length(bufs) < nb; push!(bufs, Array{TA,3}(undef, ng, ng, ng)); end
    Threads.@threads :static for b in 1:nb; fill!(bufs[b], zero(TA)); end   # first touch by the worker
    pool = Channel{Int}(nb); foreach(b -> put!(pool, b), 1:nb)
    @sync for k in 1:n
        Threads.@spawn begin
            b = take!(pool)
            _tracer_slab!(bufs[b]::Array{TA,3}, ψ, vel, acc, G, off, bary, nbary, n, dq, ng, invh, mt, mode, k)
            put!(pool, b)
        end
    end
    N = length(mesh); nt = Threads.nthreads(); ch = cld(N, nt)
    Threads.@threads :static for c in 1:nt
        r = (c - 1) * ch + 1 : min(c * ch, N)
        @inbounds for I in r
            s = zero(TA)
            for b in 1:nb; s += (bufs[b]::Array{TA,3})[I]; end
            mesh[I] = s
        end
    end
    return mesh
end

"""
    sheet_tracer_deposit_periodic(ψ, vel, L, ng, mode::Symbol; acc=nothing, G=nothing, level=1,
                                  Tacc=Float64, mesh=nothing) -> mesh (ng,ng,ng)

Conservative PCS deposit of the P1 sheet sampled by `binomial(level+3, 3)` tracers per
tetrahedron, for `mode` in `TRACER_MODES` (see there).  Derivatives act on the kernel, so
∂_t(:rho) = −(:flux) and ∂²_t(:rho) = (:stress) − (:acc) hold identically node by node.
`vel`, `acc` are (n,n,n,3) vertex fields (linearly interpolated inside each tetrahedron)."""
function sheet_tracer_deposit_periodic(ψ::AbstractArray{T,4}, vel::AbstractArray{T,4}, L::Real, ng::Int, mode::Symbol;
                                       acc=nothing, G=nothing, level::Int=1, Tacc::Type{<:AbstractFloat}=Float64,
                                       mesh=nothing) where {T}
    mi = findfirst(==(mode), TRACER_MODES); mi === nothing && error("unknown mode $mode")
    (mode === :acc && acc === nothing) && error("mode :acc needs acc")
    (mode in (:gdiv, :bulk, :disp) && G === nothing) && error("mode $mode needs G")
    n = size(ψ, 1); backend = get_backend(ψ); off = _offsets_on(ψ)
    mesh === nothing && (mesh = KernelAbstractions.zeros(backend, Tacc, ng, ng, ng))
    fill!(mesh, zero(eltype(mesh)))
    bh = sheet_tracer_bary(level); bary = similar(ψ, T, size(bh)); copyto!(bary, T.(bh))
    mt = T((L / n)^3 / (L / ng)^3 / (6 * size(bh, 1)))
    Gd = G === nothing ? reshape(mesh, ng, ng, ng, 1) : G
    if backend isa KernelAbstractions.CPU && mesh isa Array && Threads.nthreads() > 1
        _tracer_deposit_cpu!(mesh, ψ, vel, acc === nothing ? vel : acc, Gd, off, bary, size(bh, 1), n, T(L / n), ng,
                             T(ng / L), mt, mi)
    else
        _tracer_deposit!(backend)(mesh, ψ, vel, acc === nothing ? vel : acc, Gd, off, bary, size(bh, 1), n, T(L / n), ng,
                                  T(ng / L), mt, mi; ndrange=(6, n, n, n))
        synchronize(backend)
    end
    return mesh
end
