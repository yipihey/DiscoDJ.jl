"""
DiscoDJNativeCUDAExt — CUDA/cuFFT backend for DiscoDJNative.

Loaded automatically when both DiscoDJNative and CUDA are imported.  Provides
`to_gpu`/`to_host` so the *unmodified* `compute_lpt(...; backend=:ka)` runs on
the GPU: the KA kernels already infer their backend from the array, the nLPT
temporaries/outputs are allocated with `similar(fphi, ...)` (device-aware), and
the broadcast paths fuse into device kernels.

The one device-specific wrinkle is FFT layout.  CPU code keeps the rfft
half-axis on dim 3 (`rfft(x, [3,1,2])`), but cuFFT requires the reduced axis to
come first in a strictly-increasing region, i.e. dim 1.  So on the way to the
device we permute `(3,1,2)` (half-axis → dim 1) and build cuFFT plans with
region `[1,2,3]`; on the way back we undo it with the inverse permutation
`(2,3,1)`.  Everything between is elementwise/separable, so the permuted run is
numerically identical to the CPU run (validated by the GPU≈CPU test).
"""
module DiscoDJNativeCUDAExt

using DiscoDJNative
using DiscoDJNative: FourierGrid, LPTResult, HalfField, NLPTKernels, ICOperator
using CUDA
using AbstractFFTs: plan_rfft, plan_irfft, rfft, irfft

# (res,res,res÷2+1) half-on-dim3  →  device (res÷2+1,res,res) half-on-dim1.
#
# We also canonicalise the half-spectrum first.  cuFFT's C2R is stricter than
# FFTW's about the redundant modes on the DC/Nyquist planes, and the NGenIC-style
# generate_grf does not store them in canonical Hermitian form — FFTW tolerates
# this, cuFFT does not (size-dependent: fine at res 64, ~30 % wrong at 128/256).
# An irfft→rfft round-trip is idempotent under irfft (so the real-space field,
# and the whole pipeline, are physically unchanged) and yields a spectrum both
# libraries reconstruct identically.  Cost is two CPU FFTs, paid once per IC.
function DiscoDJNative.to_gpu(f::AbstractArray{Complex{T},3}) where {T}
    res  = size(f, 1)
    pinv = plan_irfft(similar(f), res, [3, 1, 2])
    pfwd = plan_rfft(Array{T}(undef, res, res, res), [3, 1, 2])
    fcanon = pfwd * (pinv * copy(f))
    return CuArray(permutedims(fcanon, (3, 1, 2)))
end

# CUDA.jl attaches a preservation buffer (one complex half-spectrum — 2.9 GiB at
# 896³) to every r2c plan, but it is only used on the inverse/ldiv! path.  The LPT
# pipeline only ever runs `plan_fwd` forward via `mul!`, so the buffer is dead
# weight: free it and leave an empty same-type stub so the plan's finalizer (which
# frees `buffer`) stays valid.
function _free_fwd_buffer!(p)
    if hasproperty(p, :buffer) && getfield(p, :buffer) !== nothing
        b = p.buffer
        if length(b) > 0
            p.buffer = similar(b, ntuple(_ -> 0, ndims(b)))
            CUDA.unsafe_free!(b)
        end
    end
    return p
end

"""
    to_gpu(grid::FourierGrid) -> FourierGrid

Device copy of a Fourier grid: k-component arrays and |k|² permuted to the
half-on-dim1 layout, with cuFFT forward/inverse plans (region `[1,2,3]`).
"""
function DiscoDJNative.to_gpu(grid::FourierGrid{T}) where {T}
    res = grid.res
    perm(x) = CuArray(permutedims(x, (3, 1, 2)))      # → (res÷2+1, res, res)
    kx, ky, kz = grid.k_vecs
    kg  = (perm(kx), perm(ky), perm(kz))
    k2g = perm(grid.k2)
    plan_fwd = _free_fwd_buffer!(plan_rfft(CUDA.zeros(T, res, res, res), [1, 2, 3]))
    plan_inv = plan_irfft(CUDA.zeros(Complex{T}, res ÷ 2 + 1, res, res), res, [1, 2, 3])
    return FourierGrid{T}(grid.dim, res, grid.boxsize, kg, k2g, plan_fwd, plan_inv)
end

# Free a device buffer eagerly (used by store=:f16 to drop the f32 source).
DiscoDJNative._free!(x::CuArray) = (CUDA.unsafe_free!(x); nothing)

# Bring a displacement field (res,res,res,3) back to host in (x,y,z) order.  A
# packed HalfField is moved compactly (the f16 residual stays f16); only the
# axis permutation and the device→host copy happen.
_unswap(a::AbstractArray{<:Any,4}) = permutedims(Array(a), (2, 3, 1, 4))
_tohost_field(a::CuArray{<:Any,4}) = _unswap(a)
_tohost_field(h::HalfField)        = HalfField(_unswap(h.dev), h.mean)
_tohost_field(::Nothing)           = nothing

DiscoDJNative.to_host(a::CuArray{<:Any,4}) = _unswap(a)
DiscoDJNative.to_host(a::CuArray)          = Array(a)
DiscoDJNative.to_host(h::HalfField)        = _tohost_field(h)

"""
    to_host(lpt::LPTResult) -> LPTResult

Copy all displacement fields of a GPU `LPTResult` to host in CPU axis order —
the canonical hand-off to the federation's injectors.  Works for both f32 and
packed-f16 (`store=:f16`) results.
"""
function DiscoDJNative.to_host(lpt::LPTResult{T}) where {T}
    return LPTResult{T}(_tohost_field(lpt.psi1),
                        _tohost_field(lpt.psi2),
                        _tohost_field(lpt.psi3),
                        lpt.n_order, lpt.res, lpt.boxsize)
end

# ── Faithful nLPT (nlpt_core) GPU path ────────────────────────────────────────
# The differentiable engine keeps the (…,…,half-on-dim3) layout on the device too;
# cuFFT only needs the reduced axis first in an increasing region, so we permute
# (3,1,2) just around the transform and undo it (2,3,1).  Numerically identical to
# the CPU `rfft(x,[3,1,2])` (validated), and all the elementwise kernels / pad-crop
# stay in the same layout — so nothing else in nlpt_core changes.
DiscoDJNative._rfftn(x::CuArray) = permutedims(rfft(permutedims(x, (3, 1, 2))), (2, 3, 1))
DiscoDJNative._irfftn(f::CuArray, n::Int) = permutedims(irfft(permutedims(f, (3, 1, 2)), n), (2, 3, 1))

# Move the faithful spectral kernels / IC map to the device (no layout change —
# the kernels broadcast against the half-on-dim3 arrays).
function DiscoDJNative.to_gpu(K::NLPTKernels{T}) where {T}
    NLPTKernels{T}(K.res, K.ext, K.boxsize,
        CuArray(K.d_dx), CuArray(K.d_dy), CuArray(K.d_dz), CuArray(K.inv_lap),
        CuArray(K.dx_ext), CuArray(K.dy_ext), CuArray(K.dz_ext))
end
DiscoDJNative.to_gpu(op::ICOperator{T}) where {T} =
    ICOperator{T, CuArray{T,3}}(CuArray(op.scale), op.res, op.boxsize)

end # module
