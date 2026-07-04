# ── Sheet query: density + deformation-tensor eigenvalues at arbitrary points ──
#
# Given the evolved tetrahedral CDM sheet (Eulerian vertex positions `x_grid`), answer
# for any query point (a "galaxy" or lightcone position): the local matter density
# (1+δ, summed over streams) and the eigenvalues of the deformation tensor F = ∂x/∂q of
# the sheet element (tetrahedron) containing it.  Builds directly on the P2 point-location
# machinery (chaining mesh + per-tet barycentric test); adds the per-tet deformation
# gradient and its (symmetric-part) eigenvalues.  CPU + CUDA via KernelAbstractions.

export sheet_query, evolve_sheet_lightcone, evolve_sheet_snapshot

# eigenvalues (descending) of a symmetric 3×3 given by its 6 independent entries.
# Analytic (Smith's trigonometric method) — branch-light, GPU-kernel-safe.
@inline function _sym_eig3(a11, a22, a33, a12, a13, a23)
    T = typeof(a11)
    p1 = a12*a12 + a13*a13 + a23*a23
    if p1 == zero(T)                                   # already diagonal
        l1 = max(a11, a22, a33); l3 = min(a11, a22, a33)
        return (l1, a11 + a22 + a33 - l1 - l3, l3)
    end
    q  = (a11 + a22 + a33) / 3
    p2 = (a11-q)^2 + (a22-q)^2 + (a33-q)^2 + 2*p1
    p  = sqrt(p2 / 6); ip = one(T) / p
    b11 = (a11-q)*ip; b22 = (a22-q)*ip; b33 = (a33-q)*ip
    b12 = a12*ip; b13 = a13*ip; b23 = a23*ip
    detB = b11*(b22*b33 - b23*b23) - b12*(b12*b33 - b23*b13) + b13*(b12*b23 - b22*b13)
    r   = clamp(detB/2, -one(T), one(T))
    phi = acos(r) / 3
    t23 = T(2.0943951023931953)                        # 2π/3
    l1  = q + 2*p*cos(phi)
    l3  = q + 2*p*cos(phi + t23)
    return (l1, 3*q - l1 - l3, l3)                     # l1 ≥ l2 ≥ l3
end

# per-tet: deformation gradient F = (1/dx)·E·M⁻¹ (E = Eulerian edge matrix, M = integer
# Lagrangian edge matrix — constant per tet type).  Compute F, its symmetric-part
# eigenvalues, and the single-stream density dx³/|det E|; scatter to contained points.
@kernel function _sheet_query_kernel!(rho, nstream, lambda, @Const(xg), @Const(off), @Const(Minv),
        @Const(pts), @Const(perm), @Const(cstart), o1, o2, o3, h, d1::Int, d2::Int, d3::Int,
        res::Int, dx, dx3, eps)
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
        c23x=e2y*e3z-e2z*e3y; c23y=e2z*e3x-e2x*e3z; c23z=e2x*e3y-e2y*e3x
        detf = e1x*c23x + e1y*c23y + e1z*c23z                            # = 6 V_T = det E
        if abs(detf) > eps
            inv  = one(detf)/detf
            dens = dx3 / abs(detf)                                       # 1+δ, single stream
            # deformation gradient F = (1/dx)·E·M⁻¹ ; E columns = e1,e2,e3
            m11=Minv[t,1,1]; m12=Minv[t,1,2]; m13=Minv[t,1,3]
            m21=Minv[t,2,1]; m22=Minv[t,2,2]; m23=Minv[t,2,3]
            m31=Minv[t,3,1]; m32=Minv[t,3,2]; m33=Minv[t,3,3]
            idx3 = one(dx)/dx
            F11=idx3*(e1x*m11+e2x*m21+e3x*m31); F12=idx3*(e1x*m12+e2x*m22+e3x*m32); F13=idx3*(e1x*m13+e2x*m23+e3x*m33)
            F21=idx3*(e1y*m11+e2y*m21+e3y*m31); F22=idx3*(e1y*m12+e2y*m22+e3y*m32); F23=idx3*(e1y*m13+e2y*m23+e3y*m33)
            F31=idx3*(e1z*m11+e2z*m21+e3z*m31); F32=idx3*(e1z*m12+e2z*m22+e3z*m32); F33=idx3*(e1z*m13+e2z*m23+e3z*m33)
            la, lb, lc = _sym_eig3(F11, F22, F33, (F12+F21)/2, (F13+F31)/2, (F23+F32)/2)
            axmn=min(y1x,y2x,y3x,y4x); axmx=max(y1x,y2x,y3x,y4x)
            aymn=min(y1y,y2y,y3y,y4y); aymx=max(y1y,y2y,y3y,y4y)
            azmn=min(y1z,y2z,y3z,y4z); azmx=max(y1z,y2z,y3z,y4z)
            cxa=clamp(unsafe_trunc(Int,(axmn-o1)/h),0,d1-1); cxb=clamp(unsafe_trunc(Int,(axmx-o1)/h),0,d1-1)
            cya=clamp(unsafe_trunc(Int,(aymn-o2)/h),0,d2-1); cyb=clamp(unsafe_trunc(Int,(aymx-o2)/h),0,d2-1)
            cza=clamp(unsafe_trunc(Int,(azmn-o3)/h),0,d3-1); czb=clamp(unsafe_trunc(Int,(azmx-o3)/h),0,d3-1)
            for cz in cza:czb, cy in cya:cyb, cx in cxa:cxb
                c = cx + d1*(cy + d2*cz) + 1
                for idxp in cstart[c]:(cstart[c+1]-1)
                    g = perm[idxp]
                    ddx=pts[g,1]-y1x; ddy=pts[g,2]-y1y; ddz=pts[g,3]-y1z
                    l2 = (ddx*c23x + ddy*c23y + ddz*c23z)*inv
                    l3 = (e1x*(ddy*e3z-ddz*e3y) + e1y*(ddz*e3x-ddx*e3z) + e1z*(ddx*e3y-ddy*e3x))*inv
                    l4 = (e1x*(e2y*ddz-e2z*ddy) + e1y*(e2z*ddx-e2x*ddz) + e1z*(e2x*ddy-e2y*ddx))*inv
                    l1 = one(l2) - l2 - l3 - l4
                    tol = oftype(l2, eps)
                    if l1 >= -tol && l2 >= -tol && l3 >= -tol && l4 >= -tol
                        KernelAbstractions.@atomic rho[g] += dens
                        KernelAbstractions.@atomic nstream[g] += Int32(1)
                        lambda[g,1]=la; lambda[g,2]=lb; lambda[g,3]=lc      # last-wins (see docstring)
                    end
                end
            end
        end
    end
end

# integer Lagrangian edge matrices M (columns q₂−q₁,q₃−q₁,q₄−q₁) per tet type → M⁻¹
function _tet_Minv(off_host)
    Minv = zeros(Float64, 6, 3, 3)
    @inbounds for t in 1:6
        M = Float64[ off_host[t,v+1,d]-off_host[t,1,d] for d in 1:3, v in 1:3 ]   # 3×3, columns = edges
        Minv[t,:,:] = inv(M)
    end
    return Minv
end

"""
    sheet_query(x_grid, pts, cl, res, boxsize; eps=1e-7) -> (; density, nstream, lambda)

For each query point (a galaxy or any lightcone position), return the local matter density
and the eigenvalues of the sheet element's deformation tensor:

- `density::(N,)` — matter density `1+δ = Σ_streams 1/|det F|` (summed over every tetrahedron
  containing the point; mean-density-normalised, so a uniform sheet gives 1).
- `nstream::(N,)` — number of sheet tetrahedra containing the point (1 single-stream, ≥3 in
  folded/multi-stream regions, 0 outside the sheet).
- `lambda::(N,3)` — the three eigenvalues (descending) of the **symmetric part of the
  deformation gradient** `F = ∂x/∂q` of the containing element — the principal stretches
  (a uniform sheet gives (1,1,1); the Zel'dovich deformation-tensor eigenvalues are `λ−1`,
  the standard void/sheet/filament/knot classifier).  `NaN` where the point is outside the sheet.

`x_grid::(res,res,res,3)` are the Eulerian sheet vertices (from `evolve_sheet_lightcone` /
`evolve_sheet_snapshot`); `cl = build_cell_list(pts, boxsize/res)`.  Detached (analysis, not on
the AD tape); runs on CPU or CUDA following `x_grid`'s backend.

For a multi-stream point `lambda` is a *representative* stream (last write wins); filter on
`nstream == 1` for an unambiguous single-stream deformation, and use `density`/`nstream` for the
multi-stream content.
"""
function sheet_query(x_grid::AbstractArray{T,4}, pts::AbstractMatrix{T}, cl, res::Int,
                     boxsize::Real; eps::Real=1e-7) where {T}
    backend = get_backend(x_grid); off = _offsets_on(x_grid)
    Minv = similar(x_grid, T, (6,3,3)); copyto!(Minv, T.(_tet_Minv(Array(off))))
    mv(x) = (y = similar(x_grid, eltype(x), size(x)); copyto!(y, x); y)
    N = size(pts,1); dx = T(boxsize/res)
    rho     = KernelAbstractions.zeros(backend, T, N)
    nstream = KernelAbstractions.zeros(backend, Int32, N)
    lambda  = similar(x_grid, T, (N,3)); fill!(lambda, T(NaN))
    _sheet_query_kernel!(backend)(rho, nstream, lambda, x_grid, off, Minv, mv(pts),
        mv(cl.perm), mv(cl.cell_start), T(cl.o1), T(cl.o2), T(cl.o3), T(cl.h),
        cl.d1, cl.d2, cl.d3, res, dx, dx^3, T(eps); ndrange=(6, res-1, res-1, res-1))
    synchronize(backend)
    return (density=rho, nstream=nstream, lambda=lambda)
end

# ── convenience: our IC (white noise ω) → evolved sheet vertices, ready to query ──

"""
    evolve_sheet_snapshot(ω, cosmo, a; boxsize, n_order=2, pk=linear_power_spectrum(cosmo))
        -> (; x_grid, res, boxsize, a)

Load a white-noise IC field `ω` (e.g. a QuaiaICs carrier realized with `refine_phases`, or a
CF4 local `ic`), evolve the tetrahedral sheet to scale factor `a` with `n_order`-LPT, and return
the Eulerian sheet vertices `x_grid::(res,res,res,3)` ready for `sheet_query`.
"""
function evolve_sheet_snapshot(ω::AbstractArray{<:Real,3}, cosmo, a::Real; boxsize::Real,
                               n_order::Int=2, pk=linear_power_spectrum(cosmo))
    res = size(ω,1); T = eltype(ω)
    op  = ic_operator(res, boxsize, pk; T=T)
    Ψ   = exact_shape_stack(compute_core_exact(white_noise_to_fphi(op, ω), nlpt_kernels(res, boxsize); n_order=n_order))
    q   = reshape(lagrangian_grid_3d(res, boxsize; T=T), res^3, 3)
    Dk  = _growth_stack(cosmo, a, size(Ψ,3))
    x   = similar(q); @inbounds for d in 1:3, n in 1:res^3
        s = q[n,d]; for kk in 1:size(Ψ,3); s += Dk[kk]*Ψ[n,d,kk]; end; x[n,d] = s; end
    return (x_grid=reshape(x, res,res,res,3), res=res, boxsize=T(boxsize), a=T(a))
end

"""
    evolve_sheet_lightcone(ω, cosmo, observer, a_far, a_near; boxsize, n_order=2, velocity=false)
        -> (; x_grid, a_cross, v_vec, valid, res, boxsize)

Evolve the sheet on the **past lightcone** of `observer`: each Lagrangian element is placed at its
own lightcone-crossing scale factor (via `lightcone_cross_ad`).  Returns the Eulerian lightcone
vertices `x_grid::(res,res,res,3)` ready for `sheet_query`, plus the per-vertex crossing scale
factor, peculiar velocity (if `velocity`), and validity.  This is the mode for querying any point
*in the lightcone*.
"""
function evolve_sheet_lightcone(ω::AbstractArray{<:Real,3}, cosmo, observer, a_far::Real, a_near::Real;
                                boxsize::Real, n_order::Int=2, velocity::Bool=false,
                                pk=linear_power_spectrum(cosmo))
    res = size(ω,1); T = eltype(ω)
    op  = ic_operator(res, boxsize, pk; T=T)
    Ψ   = exact_shape_stack(compute_core_exact(white_noise_to_fphi(op, ω), nlpt_kernels(res, boxsize); n_order=n_order))
    q   = reshape(lagrangian_grid_3d(res, boxsize; T=T), res^3, 3)
    r   = lightcone_cross_ad(Ψ, q, cosmo, collect(T, observer), T(a_far), T(a_near); velocity=velocity)
    return (x_grid=reshape(r.x_obs, res,res,res,3), a_cross=r.a_cross,
            v_vec=r.v_vec, valid=r.valid, res=res, boxsize=T(boxsize))
end

# growth factors D_k(a) for the K stacked LPT shapes (1,2,3a,3b,3c order as in exact_shape_stack)
function _growth_stack(cosmo, a, K)
    T = typeof(growth_D1(cosmo, a))
    D1 = growth_D1(cosmo, a)
    K == 1 && return (D1,)
    D2 = growth_D2(cosmo, a)
    K == 2 && return (D1, D2)
    return (D1, D2, growth_D3a(cosmo, a), growth_D3b(cosmo, a), growth_D3c(cosmo, a))
end
