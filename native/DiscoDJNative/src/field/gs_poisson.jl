"""
Differentiable red-black Gauss-Seidel Poisson smoother (periodic 7-point Laplacian).

Solves ∇²φ = δ by `nsweep` in-place red-black GS sweeps from `φ0`:

    φ_i ← (Σ_nbr φ − h2·δ_i)/6      (red cells, then black; periodic stencil, spacing²=h2)

Red-black ordering makes each colour fully parallel (a cell's 6 face-neighbours are the
*other* colour), so the sweep is one in-place KA kernel per colour — zero allocation, GPU-
ideal, "always identical". GS is the per-octave smoother: paired with a coarse FFT that
carries the long wavelengths, a handful of sweeps drives the high-frequency residual down
geometrically (the long-range 1/r part is what GS alone cannot reach).

Differentiable via a hand-written `rrule`. The map (φ0, δ) → φ is a FIXED affine operator
(the GS iteration matrix is constant — independent of the field values), so its adjoint is
just the transposed sweeps applied in reverse order, needing NO saved forward state (no tape
of the sweeps): the pullback is `nsweep` more sweeps of the same cost, constant memory. The
transpose of a colour update is `gather adjoint to the other colour, then accumulate δ̄ and
zero this colour` — the stencil is symmetric, so the gather is the same 7-point stencil.

`mask` (optional Bool field) restricts updates to the active cells — the local-patch solve
for the survey footprint; inactive cells act as Dirichlet boundaries (read, never written).
"""

export gs_smooth, laplacian7

# periodic 7-point Laplacian:  (∇²φ)_i = (Σ_nbr φ − 6φ_i)/h2  (helper for residuals / two-level)
@kernel function _lap7_kernel!(out, @Const(φ), h2, res::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        s = φ[mod1(i+1,res),j,k] + φ[mod1(i-1,res),j,k] + φ[i,mod1(j+1,res),k] +
            φ[i,mod1(j-1,res),k] + φ[i,j,mod1(k+1,res)] + φ[i,j,mod1(k-1,res)]
        out[i,j,k] = (s - 6*φ[i,j,k]) / h2
    end
end

"""    laplacian7(φ, h2) -> ∇²φ  (periodic 7-point, spacing²=h2)"""
function laplacian7(φ::AbstractArray{T,3}, h2::Real) where {T}
    backend = get_backend(φ); res = size(φ,1); out = similar(φ)
    _lap7_kernel!(backend)(out, φ, T(h2), res; ndrange=size(φ))
    synchronize(backend); return out
end

# the periodic 7-point Laplacian matrix is symmetric ⇒ self-adjoint: VJP = laplacian7(cotangent)
function ChainRulesCore.rrule(::typeof(laplacian7), φ::AbstractArray{T,3}, h2::Real) where {T}
    out = laplacian7(φ, h2)
    lap_pullback(ō) = (NoTangent(),
                       laplacian7(ō isa ChainRulesCore.AbstractZero ? zero(φ) : T.(ō), h2),
                       NoTangent())
    return out, lap_pullback
end

# ── forward: one colour update, in place ──
@kernel function _gs_color!(φ, @Const(δ), h2, color::Int, res::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds if (i+j+k) % 2 == color
        s = φ[mod1(i+1,res),j,k] + φ[mod1(i-1,res),j,k] + φ[i,mod1(j+1,res),k] +
            φ[i,mod1(j-1,res),k] + φ[i,j,mod1(k+1,res)] + φ[i,j,mod1(k-1,res)]
        φ[i,j,k] = (s - h2*δ[i,j,k]) / 6
    end
end
@kernel function _gs_color_masked!(φ, @Const(δ), h2, color::Int, res::Int, @Const(active))
    i, j, k = @index(Global, NTuple)
    @inbounds if (i+j+k) % 2 == color && active[i,j,k]
        s = φ[mod1(i+1,res),j,k] + φ[mod1(i-1,res),j,k] + φ[i,mod1(j+1,res),k] +
            φ[i,mod1(j-1,res),k] + φ[i,j,mod1(k+1,res)] + φ[i,j,mod1(k-1,res)]
        φ[i,j,k] = (s - h2*δ[i,j,k]) / 6
    end
end

# ── adjoint of a colour update: gather this colour's cotangent to the other colour (kernel1),
#    then accumulate δ̄ and zero this colour (kernel2).  Transpose of the symmetric stencil. ──
@kernel function _gs_adj_gather!(φ̄, color::Int, res::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds if (i+j+k) % 2 != color           # other-colour cells receive
        s = φ̄[mod1(i+1,res),j,k] + φ̄[mod1(i-1,res),j,k] + φ̄[i,mod1(j+1,res),k] +
            φ̄[i,mod1(j-1,res),k] + φ̄[i,j,mod1(k+1,res)] + φ̄[i,j,mod1(k-1,res)]
        φ̄[i,j,k] += s / 6
    end
end
@kernel function _gs_adj_gather_masked!(φ̄, color::Int, res::Int, @Const(active))
    i, j, k = @index(Global, NTuple)
    @inbounds if (i+j+k) % 2 != color
        s = (active[mod1(i+1,res),j,k] ? φ̄[mod1(i+1,res),j,k] : zero(eltype(φ̄))) +
            (active[mod1(i-1,res),j,k] ? φ̄[mod1(i-1,res),j,k] : zero(eltype(φ̄))) +
            (active[i,mod1(j+1,res),k] ? φ̄[i,mod1(j+1,res),k] : zero(eltype(φ̄))) +
            (active[i,mod1(j-1,res),k] ? φ̄[i,mod1(j-1,res),k] : zero(eltype(φ̄))) +
            (active[i,j,mod1(k+1,res)] ? φ̄[i,j,mod1(k+1,res)] : zero(eltype(φ̄))) +
            (active[i,j,mod1(k-1,res)] ? φ̄[i,j,mod1(k-1,res)] : zero(eltype(φ̄)))
        φ̄[i,j,k] += s / 6
    end
end
@kernel function _gs_adj_zero!(φ̄, δ̄, h2, color::Int, res::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds if (i+j+k) % 2 == color
        δ̄[i,j,k] += -(h2/6) * φ̄[i,j,k]
        φ̄[i,j,k] = 0
    end
end
@kernel function _gs_adj_zero_masked!(φ̄, δ̄, h2, color::Int, res::Int, @Const(active))
    i, j, k = @index(Global, NTuple)
    @inbounds if (i+j+k) % 2 == color && active[i,j,k]
        δ̄[i,j,k] += -(h2/6) * φ̄[i,j,k]
        φ̄[i,j,k] = 0
    end
end

# colour order: red (0) then black (1); reverse for the adjoint
_sweep!(φ, δ, h2, res, be, ::Nothing) =
    (for c in (0,1); _gs_color!(be)(φ, δ, h2, c, res; ndrange=size(φ)); synchronize(be); end)
_sweep!(φ, δ, h2, res, be, m::AbstractArray) =
    (for c in (0,1); _gs_color_masked!(be)(φ, δ, h2, c, res, m; ndrange=size(φ)); synchronize(be); end)

function _adjoint_sweep!(φ̄, δ̄, h2, res, be, ::Nothing)
    for c in (1,0)                                    # reverse of (red, black)
        _gs_adj_gather!(be)(φ̄, c, res; ndrange=size(φ̄)); synchronize(be)
        _gs_adj_zero!(be)(φ̄, δ̄, h2, c, res; ndrange=size(φ̄)); synchronize(be)
    end
end
function _adjoint_sweep!(φ̄, δ̄, h2, res, be, m::AbstractArray)
    for c in (1,0)
        _gs_adj_gather_masked!(be)(φ̄, c, res, m; ndrange=size(φ̄)); synchronize(be)
        _gs_adj_zero_masked!(be)(φ̄, δ̄, h2, c, res, m; ndrange=size(φ̄)); synchronize(be)
    end
end

"""
    gs_smooth(φ0, δ, h2, nsweep; mask=nothing) -> φ

`nsweep` red-black GS sweeps of ∇²φ=δ from `φ0` (periodic 7-point, spacing²=`h2`).
Differentiable w.r.t. `φ0` and `δ` (reverse-sweep adjoint, constant memory). With `mask`
(Bool, size of φ) only the active cells are updated — the local-patch solve.
"""
function gs_smooth(φ0::AbstractArray{T,3}, δ::AbstractArray{T,3}, h2::Real, nsweep::Int;
                   mask=nothing) where {T}
    backend = get_backend(φ0); res = size(φ0,1); h2T = T(h2)
    φ = copy(φ0)
    for _ in 1:nsweep; _sweep!(φ, δ, h2T, res, backend, mask); end
    return φ
end

function ChainRulesCore.rrule(::typeof(gs_smooth), φ0::AbstractArray{T,3}, δ::AbstractArray{T,3},
                              h2::Real, nsweep::Int; mask=nothing) where {T}
    φ = gs_smooth(φ0, δ, h2, nsweep; mask)
    function gs_pullback(Δ)
        backend = get_backend(φ0); res = size(φ0,1); h2T = T(h2)
        φ̄ = Δ isa ChainRulesCore.AbstractZero ? KernelAbstractions.zeros(backend, T, size(φ0)) : copy(T.(Δ))
        δ̄ = KernelAbstractions.zeros(backend, T, size(δ))
        for _ in 1:nsweep; _adjoint_sweep!(φ̄, δ̄, h2T, res, backend, mask); end
        return (NoTangent(), φ̄, δ̄, NoTangent(), NoTangent())
    end
    return φ, gs_pullback
end
