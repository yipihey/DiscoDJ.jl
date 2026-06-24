"""
Compact half-precision storage for LPT displacement fields.

The nLPT displacement components have a modest dynamic range, so after removing a
per-component mean (kept in Float32) the residual fits comfortably in Float16.
That halves the footprint of the `(res,res,res,3)` displacement fields — 12 →
6 bytes/cell — letting a substantially larger box's displacements live in GPU
memory.  Device-aware: `dev` is an `Array` on the CPU, a `CuArray` on the GPU.

The Float16 round-off (~5e-4 relative) is well below LPT's own accuracy and
comparable to the existing GPU Nyquist residual, so it is harmless for ICs.
"""

export HalfField, pack_half, expand_half

struct HalfField{A<:AbstractArray{Float16,4}}
    dev::A                      # Float16 deviations, (res,res,res,3)
    mean::NTuple{3,Float32}     # per-component mean, removed before the f16 cast
end

resolution(h::HalfField) = size(h.dev, 1)
Base.eltype(::HalfField) = Float16
Base.size(h::HalfField)  = size(h.dev)

"""
    pack_half(A) -> HalfField

Pack a real `(res,res,res,3)` displacement field: subtract each component's mean
(stored Float32) and cast the residual to Float16.  Runs on the array's device.
"""
function pack_half(A::AbstractArray{<:Real,4})
    @assert size(A, 4) == 3 "expected a (res,res,res,3) displacement field"
    ncell = size(A, 1) * size(A, 2) * size(A, 3)
    m = ntuple(d -> Float32(sum(@view A[:, :, :, d]) / ncell), 3)
    dev = similar(A, Float16)
    for d in 1:3
        @views dev[:, :, :, d] .= Float16.(A[:, :, :, d] .- m[d])
    end
    return HalfField(dev, m)
end

"""
    expand_half(h) -> Array{Float32,4}

Reconstruct the Float32 field from a `HalfField`, on the same device as `h.dev`.
"""
function expand_half(h::HalfField)
    A = similar(h.dev, Float32)
    for d in 1:3
        @views A[:, :, :, d] .= Float32.(h.dev[:, :, :, d]) .+ h.mean[d]
    end
    return A
end

Base.Array(h::HalfField) = Array(expand_half(h))
