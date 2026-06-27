#=
Masked AHK nodal density — the SHEET-ON-MASK local-patch solve.

The tetrahedral sheet density restricted to the survey footprint's Lagrangian trace-back `active`
mask (a Bool field on the vertices, including a ≥1-cell buffer).  A tet is processed iff its anchor
vertex (i,j,k) is active; inactive tets are skipped — only ~10% of the 6·res³ tets (the footprint)
are touched.

Exactness: the survey window already folds W=0 into `w`, so every skipped tet has w_T=0 and would
contribute 0 to N_v and Z; the only quantity the skip changes is D_v (the geometric volume sum) at
buffer vertices, but there N_v=0 ⇒ ρ_v=N_v/D_v=0 regardless.  With the buffer, every footprint
vertex has all its incident tets active ⇒ ρ_v there and Z are BYTE-for-byte the full windowed result.
Every operator is the same as the unmasked kernel plus an `active[i,j,k]` gate ⇒ same differentiability.
=#
export nodal_density_masked

@kernel function _nodal_fwd_masked!(Nv, Dv, Z, @Const(xg), @Const(wg), @Const(off), res::Int, mT, floorvol, @Const(active))
    t, i, j, k = @index(Global, NTuple)
    @inbounds if active[i,j,k]
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

@kernel function _nodal_bwd_masked!(x̄, w̄, @Const(N̄v), @Const(D̄v), Z̄, @Const(xg), @Const(off), res::Int, mT, floorvol, @Const(active))
    t, i, j, k = @index(Global, NTuple)
    @inbounds if active[i,j,k]
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

"""    nodal_density_masked(x_grid, w, res, boxsize, active; floor_frac=1e-3) -> (ρ_v, Z)

AHK nodal density restricted to the `active` (Bool res³) footprint trace-back mask — only tets whose
anchor vertex is active are processed.  Exact at footprint vertices (see file header).  Differentiable
w.r.t. `x_grid`, `w`."""
function nodal_density_masked(x_grid::AbstractArray{T,4}, w::AbstractArray{T,3}, res::Int,
                              boxsize::Real, active::AbstractArray{Bool,3}; floor_frac::Real=1e-3) where {T}
    backend = get_backend(x_grid); off = _offsets_on(x_grid)
    mT = T(1//6); floorvol = T(floor_frac)*(T(boxsize)/res)^3/6
    Nv = KernelAbstractions.zeros(backend, Float64, res, res, res); Dv = KernelAbstractions.zeros(backend, Float64, res, res, res)
    Z = KernelAbstractions.zeros(backend, Float64, 1)
    _nodal_fwd_masked!(backend)(Nv, Dv, Z, x_grid, w, off, res, mT, floorvol, active; ndrange=(6,res-1,res-1,res-1))
    synchronize(backend)
    Dc = max.(Dv, Float64(floorvol)); ρv = T.(Nv ./ Dc)
    return (ρv, Array(Z)[1])
end

function ChainRulesCore.rrule(::typeof(nodal_density_masked), x_grid::AbstractArray{T,4},
                              w::AbstractArray{T,3}, res::Int, boxsize::Real, active::AbstractArray{Bool,3};
                              floor_frac::Real=1e-3) where {T}
    backend = get_backend(x_grid); off = _offsets_on(x_grid)
    mT = T(1//6); floorvol = T(floor_frac)*(T(boxsize)/res)^3/6
    Nv = KernelAbstractions.zeros(backend, Float64, res, res, res); Dv = KernelAbstractions.zeros(backend, Float64, res, res, res)
    Z = KernelAbstractions.zeros(backend, Float64, 1)
    _nodal_fwd_masked!(backend)(Nv, Dv, Z, x_grid, w, off, res, mT, floorvol, active; ndrange=(6,res-1,res-1,res-1))
    synchronize(backend)
    Dc = max.(Dv, Float64(floorvol)); ρv = T.(Nv ./ Dc); Zv = Array(Z)[1]
    function nodal_masked_pullback(Δ)
        ρ̄v = Δ[1] isa ChainRulesCore.AbstractZero ? KernelAbstractions.zeros(backend, Float64, res,res,res) :
             (y=KernelAbstractions.zeros(backend,Float64,res,res,res); copyto!(y, Float64.(unthunk(Δ[1]))); y)
        Z̄ = Δ[2] isa ChainRulesCore.AbstractZero ? 0.0 : Float64(Δ[2])
        N̄v = ρ̄v ./ Dc
        D̄v = @. -ρ̄v * Nv / (Dc*Dc) * (Dv > Float64(floorvol))
        x̄ = KernelAbstractions.zeros(backend, Float64, res,res,res,3); w̄ = KernelAbstractions.zeros(backend, Float64, res,res,res)
        _nodal_bwd_masked!(backend)(x̄, w̄, N̄v, D̄v, Z̄, x_grid, off, res, mT, floorvol, active; ndrange=(6,res-1,res-1,res-1))
        synchronize(backend)
        return (NoTangent(), T.(x̄), T.(w̄), NoTangent(), NoTangent(), NoTangent())
    end
    return (ρv, Zv), nodal_masked_pullback
end

# ── masked C⁰ barycentric interpolation at the galaxies (ρ_g) ──────────────────
export interp_sheet_at_points_masked

@kernel function _interp_fwd_masked!(ρg, @Const(xg), @Const(ρv), @Const(off), @Const(pts), @Const(perm),
        @Const(cstart), o1,o2,o3,h,d1::Int,d2::Int,d3::Int, res::Int, eps, @Const(active))
    t,i,j,k = @index(Global, NTuple)
    @inbounds if active[i,j,k]
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

@kernel function _interp_bwd_masked!(x̄, ρ̄v, @Const(ρ̄g), @Const(xg), @Const(ρv), @Const(off), @Const(pts),
        @Const(perm), @Const(cstart), o1,o2,o3,h,d1::Int,d2::Int,d3::Int, res::Int, eps, @Const(active))
    t,i,j,k = @index(Global, NTuple)
    @inbounds if active[i,j,k]
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
            gρx=((ρ2-ρ1)*c23x+(ρ3-ρ1)*c31x+(ρ4-ρ1)*c12x)*inv
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
                        b1=-rb*l1;b2=-rb*l2;b3=-rb*l3;b4=-rb*l4
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

"""    interp_sheet_at_points_masked(x_grid, ρ_v, pts, cl, res, active; eps=1e-7) -> ρ_g

C⁰ barycentric density at the galaxies, restricted to the `active` footprint mask (only active tets
search for their galaxies).  Galaxies sit in the footprint ⇒ their containing tet is active ⇒ ρ_g is
exact.  Differentiable w.r.t. `x_grid`, `ρ_v`."""
function interp_sheet_at_points_masked(x_grid::AbstractArray{T,4}, ρv::AbstractArray{T,3},
        pts::AbstractMatrix{T}, cl, res::Int, active::AbstractArray{Bool,3}; eps::Real=1e-7) where {T}
    backend = get_backend(x_grid); off = _offsets_on(x_grid)
    mv(x)=(y=similar(x_grid,eltype(x),size(x)); copyto!(y,x); y)
    ρg = KernelAbstractions.zeros(backend, Float64, size(pts,1))
    _interp_fwd_masked!(backend)(ρg, x_grid, ρv, off, mv(pts), mv(cl.perm), mv(cl.cell_start),
        T(cl.o1),T(cl.o2),T(cl.o3),T(cl.h),cl.d1,cl.d2,cl.d3, res, T(eps), active; ndrange=(6,res-1,res-1,res-1))
    synchronize(backend)
    return ρg
end

function ChainRulesCore.rrule(::typeof(interp_sheet_at_points_masked), x_grid::AbstractArray{T,4},
        ρv::AbstractArray{T,3}, pts::AbstractMatrix{T}, cl, res::Int, active::AbstractArray{Bool,3};
        eps::Real=1e-7) where {T}
    ρg = interp_sheet_at_points_masked(x_grid, ρv, pts, cl, res, active; eps)
    backend = get_backend(x_grid); off = _offsets_on(x_grid)
    mv(x)=(y=similar(x_grid,eltype(x),size(x)); copyto!(y,x); y)
    ptsb=mv(pts); permb=mv(cl.perm); cstartb=mv(cl.cell_start)
    function interp_masked_pullback(Δ)
        ρ̄g = Δ isa ChainRulesCore.AbstractZero ? KernelAbstractions.zeros(backend,Float64,size(pts,1)) :
             (y=KernelAbstractions.zeros(backend,Float64,size(pts,1)); copyto!(y, Float64.(unthunk(Δ))); y)
        x̄ = KernelAbstractions.zeros(backend, Float64, res,res,res,3); ρ̄v = KernelAbstractions.zeros(backend, Float64, res,res,res)
        _interp_bwd_masked!(backend)(x̄, ρ̄v, ρ̄g, x_grid, ρv, off, ptsb, permb, cstartb,
            T(cl.o1),T(cl.o2),T(cl.o3),T(cl.h),cl.d1,cl.d2,cl.d3, res, T(eps), active; ndrange=(6,res-1,res-1,res-1))
        synchronize(backend)
        return (NoTangent(), T.(x̄), T.(ρ̄v), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end
    return ρg, interp_masked_pullback
end
