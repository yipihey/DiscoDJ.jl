"""
Fourier-grid utilities for LPT computations.

Provides k-vectors, |k|² arrays, Hermitian symmetry enforcement,
and the rfft ↔ irfft wrappers used throughout the LPT pipeline.
All arrays are real-layout (rfft) to save memory and match JAX convention.
"""

export FourierGrid, get_fourier_grid, fourier_gradient!, fourier_poisson!,
       canonicalize_hermitian

using FFTW
using AbstractFFTs

# ── FFTW planning policy ──────────────────────────────────────────────────────
# Default to ESTIMATE.  MEASURE/PATIENT give ~25-30 % faster transforms, but on
# this stack (FFTW.jl's Julia-`@spawn` threading backend, Julia 1.12) the extra
# threaded execution they do *during planning* intermittently segfaults inside
# FFTW's `spawnloop` — ESTIMATE is race-free.  MEASURE is therefore opt-in via
# DISCODJ_FFTW_PLANNER=measure (results are cached on disk as FFTW wisdom so the
# planning search is paid once).  Most of MEASURE's win is also recoverable
# safely just by giving the FFT more threads (ESTIMATE@64 ≈ MEASURE@16 here).
const _PLANNER_FLAGS = Dict("estimate"=>FFTW.ESTIMATE, "measure"=>FFTW.MEASURE,
                            "patient"=>FFTW.PATIENT, "exhaustive"=>FFTW.EXHAUSTIVE)
default_fftw_planner() = get(_PLANNER_FLAGS, lowercase(get(ENV, "DISCODJ_FFTW_PLANNER", "estimate")), FFTW.ESTIMATE)

# Wisdom is keyed by thread count (FFTW plans differ per nthreads).
_wisdom_file() = joinpath(get(ENV, "DISCODJ_CACHE", joinpath(homedir(), ".cache", "discodj")),
                          "fftw_wisdom_t$(FFTW.get_num_threads()).dat")
function _load_fftw_wisdom()
    f = _wisdom_file()
    isfile(f) && try; FFTW.import_wisdom(f); catch; end
    return nothing
end
function _save_fftw_wisdom()
    f = _wisdom_file()
    try; mkpath(dirname(f)); FFTW.export_wisdom(f); catch; end
    return nothing
end

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
    k_vecs::NTuple{3, AbstractArray{T, 3}}  # (kx,ky,kz); Array on CPU, CuArray on GPU
    k2::AbstractArray{T, 3}
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
function get_fourier_grid(res::Int, boxsize::Real; T::Type{<:AbstractFloat}=Float64,
                          planner=default_fftw_planner())
    dk = T(2π / boxsize)
    # Full-period k values: 0, 1, ..., N/2, -(N/2-1), ..., -1
    kfull = [i <= res÷2 ? T(i) : T(i - res) for i in 0:res-1] .* dk
    khalf = collect(T, 0:res÷2) .* dk

    # Keep the k-components as compact, separable reshaped 1-D vectors — they
    # broadcast to the full (res,res,res÷2+1) rfft shape on use, so we never
    # materialise three dense k-arrays (≈4.3 GiB at 896³) and the elementwise
    # kernels read O(res) instead of O(res³) k-values.  k² is materialised dense
    # (used by inv_laplace and asserted dense by `get_fourier_grid` callers/tests).
    kx = collect(reshape(kfull, res, 1, 1))
    ky = collect(reshape(kfull, 1, res, 1))
    kz = collect(reshape(khalf, 1, 1, res÷2+1))

    k2 = @. kx^2 + ky^2 + kz^2

    # Build plans once; all subsequent rfft/irfft calls in nlpt.jl reuse these.
    # MEASURE/PATIENT clobber their input arrays during planning, so plan against
    # throwaway scratch.  Load/save disk wisdom around planning (only meaningful
    # for the measuring planners) so the search is paid only once per size.
    measuring = planner != FFTW.ESTIMATE
    measuring && _load_fftw_wisdom()
    plan_fwd = plan_rfft(zeros(T, res, res, res), [3, 1, 2]; flags=planner)
    plan_inv = plan_irfft(zeros(Complex{T}, res, res, res÷2+1), res, [3, 1, 2]; flags=planner)
    measuring && _save_fftw_wisdom()

    return FourierGrid{T}(3, res, T(boxsize), (kx, ky, kz), k2, plan_fwd, plan_inv)
end

# ── Hermitian canonicalisation ────────────────────────────────────────────────

"""
    canonicalize_hermitian(fphi, grid) -> canonical half-spectrum

Round-trip `fphi` through irfft→rfft (using the CPU `grid`'s cached plans) so the
redundant modes on the DC/Nyquist planes are stored in canonical Hermitian form.

This is idempotent under irfft — the real-space field is unchanged, so the LPT
physics is unchanged — but the gradient/source fields the pipeline derives by
multiplying with `i·k`/`-kᵢkⱼ` differ on those boundary modes between a canonical
and a non-canonical spectrum.  `generate_grf` (NGenIC-style) does *not* produce a
canonical spectrum; FFTW's C2R tolerates that, cuFFT's does not.  Feed the result
of this function to *both* CPU and GPU runs to get bit-for-bit agreement.
"""
canonicalize_hermitian(fphi::AbstractArray{<:Complex}, grid::FourierGrid) =
    grid.plan_fwd * (grid.plan_inv * copy(fphi))

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
