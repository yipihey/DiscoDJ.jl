"""
Fourier-grid utilities for LPT computations.

Provides k-vectors, |k|² arrays, Hermitian symmetry enforcement,
and the rfft ↔ irfft wrappers used throughout the LPT pipeline.
All arrays are real-layout (rfft) to save memory and match JAX convention.
"""

export FourierGrid, get_fourier_grid, fourier_gradient!, fourier_poisson!

using FFTW
using AbstractFFTs

# ── FourierGrid ───────────────────────────────────────────────────────────────

"""
    FourierGrid{T}

Pre-computed Fourier-space grid quantities for a `dim`-dimensional periodic box.
Fields:
- `k_vecs`:  d×n_modes array of k_x, k_y, k_z components (units: h/Mpc)
- `k2`:      |k|² array, shape of the rfft output
- `kx,ky,kz`: individual k-component arrays broadcast-compatible with rfft shape
- `res`, `boxsize`, `dim`
"""
struct FourierGrid{T<:AbstractFloat}
    dim::Int
    res::Int
    boxsize::T
    k_vecs::NTuple{3, Array{T, 3}}  # (kx, ky, kz) each res×res×(res÷2+1)
    k2::Array{T, 3}
    # Cached FFTW plans — reused across all LPT calls; replace bare rfft/irfft
    # calls throughout nlpt.jl.  Stored as Any so the same struct works with
    # FFTW on CPU and cuFFT / rocFFT on GPU (via AbstractFFTs.plan_* API).
    plan_fwd::Any   # plan_rfft:  (res,res,res){T}       → (res,res,res÷2+1){Complex{T}}
    plan_inv::Any   # plan_irfft: (res,res,res÷2+1){Complex{T}} → (res,res,res){T}
end

"""
    get_fourier_grid(res, boxsize; T=Float64) -> FourierGrid

Build the 3D Fourier grid for a cubic box of side `boxsize` [Mpc/h] and
resolution `res` (particles per side). Only 3D is supported (dim=3).
"""
function get_fourier_grid(res::Int, boxsize::Real; T::Type{<:AbstractFloat}=Float64)
    dk = T(2π / boxsize)
    # Full-period k values: 0, 1, ..., N/2, -(N/2-1), ..., -1
    kfull = [i <= res÷2 ? T(i) : T(i - res) for i in 0:res-1] .* dk
    khalf = collect(T, 0:res÷2) .* dk

    # Broadcast to 3D rfft shape (res, res, res÷2+1)
    kx = reshape(kfull, res, 1, 1) .* ones(T, 1, res, res÷2+1)
    ky = reshape(kfull, 1, res, 1) .* ones(T, res, 1, res÷2+1)
    kz = reshape(khalf, 1, 1, res÷2+1) .* ones(T, res, res, 1)

    k2 = @. kx^2 + ky^2 + kz^2

    # Build plans once; all subsequent rfft/irfft calls in nlpt.jl use these.
    # plan_rfft/plan_irfft touch only shape+type of the dummy arrays.
    plan_fwd = plan_rfft(zeros(T, res, res, res), [3, 1, 2])
    plan_inv = plan_irfft(zeros(Complex{T}, res, res, res÷2+1), res, [3, 1, 2])

    return FourierGrid{T}(3, res, T(boxsize), (kx, ky, kz), k2, plan_fwd, plan_inv)
end

# ── Fourier-space operations ──────────────────────────────────────────────────

"""
    fourier_gradient!(out, fphi, k_component)

Multiply Fourier-space field fphi by i·k_component → Fourier-space gradient.
`out` is overwritten in-place. Works on any array type (CPU or GPU).
"""
function fourier_gradient!(out::AbstractArray{Complex{T}},
                           fphi::AbstractArray{Complex{T}},
                           k_comp::AbstractArray{T}) where T
    @inbounds for i in eachindex(out)
        out[i] = im * k_comp[i] * fphi[i]
    end
    return out
end

"""
    fourier_poisson!(out, fdelta, k2)

Solve the Poisson equation in Fourier space: φ(k) = -δ(k) / k².
k=0 mode is set to zero (mean displacement = 0).
"""
function fourier_poisson!(out::AbstractArray{Complex{T}},
                          fdelta::AbstractArray{Complex{T}},
                          k2::AbstractArray{T}) where T
    @inbounds for i in eachindex(out)
        k2i = k2[i]
        out[i] = k2i == 0 ? zero(Complex{T}) : -fdelta[i] / k2i
    end
    return out
end

"""
    enforce_hermitian_symmetry!(f, res)

Ensure the 3D rfft field has Hermitian symmetry at the ix=0 plane
(needed after operations that may break it).
"""
function enforce_hermitian_symmetry!(f::AbstractArray{Complex{T}, 3}, res::Int) where T
    for iy in 1:res, iz in 1:res÷2+1
        iy2 = iy == 1 ? 1 : res - iy + 2
        c   = f[1, iy, iz]
        c2  = f[1, iy2, iz]
        v   = (c + conj(c2)) / 2
        f[1, iy, iz]  = v
        f[1, iy2, iz] = conj(v)
    end
    return f
end
