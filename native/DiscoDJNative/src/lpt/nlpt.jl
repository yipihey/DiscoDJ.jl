"""
nLPT (n-th order Lagrangian Perturbation Theory) displacement field computation.

Implements 1LPT (Zel'dovich), 2LPT, and 3LPT in 3D following
arXiv:2010.12584 (List, Hahn, Winkler & Floess).

Entry point: `compute_lpt(fphi_ini, grid; n_order=2, backend=:ka)`

Returns a Dict with keys :psi1, :psi2, :psi3 — each a (res,res,res,3)
real array of displacement vectors in Mpc/h (un-normalised; apply D(a)·ψ
to get physical displacements at scale factor a).

FFT convention: arrays are (res, res, res÷2+1) in full 3D Fourier space with
the half-complex axis last. Use rfft(x, [3,1,2]) / irfft(x, res, [3,1,2]).
generate_grf returns the potential φ(k) = δ(k)/k² directly; LPT uses it
without another Poisson solve.
"""

export compute_lpt, LPTResult

using FFTW
using LinearAlgebra: mul!

# ── Portable Fourier-space second derivative  fd = -k_i k_j φ ──────────────────
# Built O(10) times per nLPT call, so it is a real cost at large res.  Multiple
# dispatch keeps one source of truth across backends: a threaded loop on host
# Arrays, and a fused broadcast (a single GPU kernel) on device arrays — so the
# unmodified nLPT code below runs on both CPU and GPU.
@inline function _build_second_deriv!(fd::Array, ki, kj, fphi)
    Threads.@threads for idx in eachindex(fd)
        @inbounds fd[idx] = -ki[idx] * kj[idx] * fphi[idx]
    end
    return fd
end
@inline _build_second_deriv!(fd::AbstractArray, ki, kj, fphi) = (@. fd = -ki * kj * fphi)

# ── Result type ───────────────────────────────────────────────────────────────

"""
    LPTResult

Holds displacement fields from nLPT computation.
- `psi1`: 1LPT (Zel'dovich) displacement  (res,res,res,3)  [Mpc/h]
- `psi2`: 2LPT correction                 (res,res,res,3)  [Mpc/h], or nothing
- `psi3`: 3LPT correction                 (res,res,res,3)  [Mpc/h], or nothing
"""
struct LPTResult{T}
    psi1::Union{AbstractArray{T, 4}, HalfField}     # Array/CuArray, or packed f16
    psi2::Union{AbstractArray{T, 4}, HalfField, Nothing}
    psi3::Union{AbstractArray{T, 4}, HalfField, Nothing}
    n_order::Int
    res::Int
    boxsize::T
end

# Optionally pack a freshly-computed f32 displacement into compact f16 storage,
# eagerly freeing the f32 source so the GPU peak drops (see `store` in compute_lpt).
function _maybe_pack(psi, store::Symbol)
    store === :f16 || return psi
    h = pack_half(psi)
    _free!(psi)
    return h
end

# ── Kernel dispatch ───────────────────────────────────────────────────────────

function _inv_laplace!(out, f, k2; backend)
    if backend == :ka
        inv_laplace_ka!(out, f, k2)
    else
        inv_laplace_threads!(out, f, k2)
    end
end

function _grad_multiply!(out, fphi, kcomp; backend)
    if backend == :ka
        grad_multiply_ka!(out, fphi, kcomp)
    else
        grad_multiply_threads!(out, fphi, kcomp)
    end
end

function _fmu2!(out, f1, f2; backend)
    if backend == :ka
        fmu2_elementwise_ka!(out, f1, f2)
    else
        fmu2_elementwise_threads!(out, f1, f2)
    end
end

# ── shared building blocks (buffer-reusing) ───────────────────────────────────

# Second derivative φ,ij in real space, written into the preallocated real `buf`
# using the complex scratch `fd`: fd = -kᵢkⱼ·φ, buf = irfft(fd).  `mul!` keeps the
# transform out of the allocator; the c2r destroys fd, which is rebuilt each call.
function _sd_into!(buf, fd, ki, kj, fphi, grid)
    _build_second_deriv!(fd, ki, kj, fphi)
    mul!(buf, grid.plan_inv, fd)
    return buf
end

# ψ_d = irfft(i·k_d·φ) for d=x,y,z, written into `psi[:,:,:,d]`, reusing the real
# buffer `rbuf` and complex scratch `fd` (no per-component allocation).
function _grad_to_psi!(psi, fphi, grid, rbuf, fd; backend)
    for (d, kc) in enumerate(grid.k_vecs)
        _grad_multiply!(fd, fphi, kc; backend)
        mul!(rbuf, grid.plan_inv, fd)
        @views psi[:, :, :, d] .= rbuf
    end
    return psi
end

# ── 1LPT ─────────────────────────────────────────────────────────────────────

function _compute_1lpt(fphi_ini::AbstractArray{Complex{T}}, grid::FourierGrid{T};
                       backend=:ka) where T
    res = grid.res
    fphi1 = fphi_ini    # already φ₁(k)=δ/k² from generate_grf (Poisson pre-solved)
    psi1 = similar(fphi1, T, res, res, res, 3)
    fd   = similar(fphi1)
    rbuf = similar(fphi1, T, res, res, res)
    _grad_to_psi!(psi1, fphi1, grid, rbuf, fd; backend)
    return psi1, fphi1
end

# ── 2LPT ─────────────────────────────────────────────────────────────────────
#
# S₂ = Σ_{i<j}(φ,ii φ,jj − φ,ij²).  Using the trace identity
#   S₂ = ½[(tr H)² − tr(H²)],  tr H = Σ d_ii,  tr(H²) = Σ d_ii² + 2 Σ_{i<j} d_ij²,
# the source accumulates from one second-derivative at a time — 3 real buffers
# (t = trace→S₂, q = tr(H²), tmp = current d_ij) instead of materialising all six.

function _compute_2lpt(fphi1::AbstractArray{Complex{T}}, grid::FourierGrid{T};
                       backend=:ka) where T
    kx, ky, kz = grid.k_vecs
    k2 = grid.k2
    res = grid.res

    fd  = similar(fphi1)                       # complex scratch (also reused as fS2)
    t   = similar(fphi1, T, res, res, res)     # Σ d_ii   → overwritten with S₂
    q   = similar(t)                           # tr(H²)
    tmp = similar(t)                           # one d_ij at a time
    sd!(ki, kj) = _sd_into!(tmp, fd, ki, kj, fphi1, grid)

    sd!(kx, kx); @. t  = tmp;          @. q  = tmp*tmp     # d11
    sd!(ky, ky); @. t += tmp;          @. q += tmp*tmp     # d22
    sd!(kz, kz); @. t += tmp;          @. q += tmp*tmp     # d33
    sd!(kx, ky); @. q += T(2)*tmp*tmp                      # d12
    sd!(kx, kz); @. q += T(2)*tmp*tmp                      # d13
    sd!(ky, kz); @. q += T(2)*tmp*tmp                      # d23
    @. t = T(0.5) * (t*t - q)                              # S₂ = ½[(trH)² − tr(H²)]

    mul!(fd, grid.plan_fwd, t)                 # fS2 = rfft(S₂) into reused scratch
    fphi2 = similar(fphi1)
    _inv_laplace!(fphi2, fd, k2; backend)
    fphi2 .*= T(-3/7)

    psi2 = similar(fphi1, T, res, res, res, 3)
    _grad_to_psi!(psi2, fphi2, grid, t, fd; backend)       # reuse t (real), fd (complex)
    return psi2, fphi2
end

# ── 3LPT ─────────────────────────────────────────────────────────────────────
#
# ∇²φ₃ = (10/21)·S₃ᵃ + (1/3)·S₃ᵇ,  S₃ᵃ = det(φ₁,ij),
#   S₃ᵇ = Σ φ₂,ii φ₁,jj − 2 Σ_{i<j} φ₂,ij φ₁,ij.
# det needs all six first-order second derivatives at once, but each second-order
# derivative φ₂,ij enters S₃ᵇ exactly once — so we hold the six d1 and stream the
# six d2 one at a time: 8 real buffers instead of 14.

function _compute_3lpt(fphi1::AbstractArray{Complex{T}}, fphi2::AbstractArray{Complex{T}},
                       grid::FourierGrid{T}; backend=:ka) where T
    kx, ky, kz = grid.k_vecs
    k2 = grid.k2
    res = grid.res

    fd = similar(fphi1)
    mk() = similar(fphi1, T, res, res, res)
    d11, d22, d33 = mk(), mk(), mk()
    d12, d13, d23 = mk(), mk(), mk()
    S3, tmp = mk(), mk()

    _sd_into!(d11, fd, kx, kx, fphi1, grid); _sd_into!(d22, fd, ky, ky, fphi1, grid)
    _sd_into!(d33, fd, kz, kz, fphi1, grid); _sd_into!(d12, fd, kx, ky, fphi1, grid)
    _sd_into!(d13, fd, kx, kz, fphi1, grid); _sd_into!(d23, fd, ky, kz, fphi1, grid)

    # S₃ = (10/21)·det(H₁)
    @. S3 = T(10/21) * (d11*(d22*d33 - d23*d23) -
                        d12*(d12*d33 - d23*d13) +
                        d13*(d12*d23 - d22*d13))

    # + (1/3)·S₃ᵇ, streaming one d2 at a time into tmp
    sd2!(ki, kj) = _sd_into!(tmp, fd, ki, kj, fphi2, grid)
    sd2!(kx, kx); @. S3 += T(1/3)*tmp*(d22 + d33)     # φ₂,11
    sd2!(ky, ky); @. S3 += T(1/3)*tmp*(d11 + d33)     # φ₂,22
    sd2!(kz, kz); @. S3 += T(1/3)*tmp*(d11 + d22)     # φ₂,33
    sd2!(kx, ky); @. S3 -= T(2/3)*tmp*d12             # φ₂,12
    sd2!(kx, kz); @. S3 -= T(2/3)*tmp*d13             # φ₂,13
    sd2!(ky, kz); @. S3 -= T(2/3)*tmp*d23             # φ₂,23

    mul!(fd, grid.plan_fwd, S3)                # fS3 into reused scratch
    fphi3 = similar(fphi1)
    _inv_laplace!(fphi3, fd, k2; backend)
    fphi3 .*= T(-1)

    psi3 = similar(fphi1, T, res, res, res, 3)
    _grad_to_psi!(psi3, fphi3, grid, tmp, fd; backend)     # reuse tmp (real), fd (complex)
    return psi3
end

# ── Main entry point ──────────────────────────────────────────────────────────

"""
    compute_lpt(fphi_ini, grid; n_order=2, backend=:ka) -> LPTResult

Compute nLPT displacement fields from the initial Fourier-space potential.

`fphi_ini` — complex rfft array of shape (res, res, res÷2+1), the Gaussian
             random potential field φ(k) = δ(k)/k² (from `generate_grf`).
`grid`     — FourierGrid for the simulation box.
`n_order`  — LPT order: 1 (Zel'dovich), 2, or 3.
`backend`  — `:ka` (KernelAbstractions) or `:threads` (Threads.@threads).

Returns `LPTResult` with un-normalised displacements ψ₁, ψ₂, ψ₃.
To get physical displacements at scale factor a:
    ψ(a) = D₁(a)·ψ₁ + D₂(a)·D₁(a=1)²·ψ₂ + …

`store` controls how the displacement fields are kept: `:f32` (default) or `:f16`
— the latter packs each ψ into a [`HalfField`](@ref) (per-component f32 mean +
f16 residual) as soon as it is computed, freeing the f32 source.  That halves the
displacement footprint and, because the f32 buffers are released eagerly, lets a
larger box fit in GPU memory.
"""
function compute_lpt(fphi_ini::AbstractArray{Complex{T}}, grid::FourierGrid{T};
                     n_order::Int=2, backend::Symbol=:ka, store::Symbol=:f32) where T
    n_order in (1, 2, 3) || error("n_order must be 1, 2, or 3")
    store in (:f32, :f16) || error("store must be :f32 or :f16")
    res = grid.res

    psi1, fphi1 = _compute_1lpt(fphi_ini, grid; backend)
    psi1 = _maybe_pack(psi1, store)
    psi2 = nothing; fphi2 = nothing
    psi3 = nothing

    if n_order >= 2
        psi2, fphi2 = _compute_2lpt(fphi1, grid; backend)
        psi2 = _maybe_pack(psi2, store)
    end
    if n_order >= 3
        psi3 = _compute_3lpt(fphi1, fphi2, grid; backend)
        psi3 = _maybe_pack(psi3, store)
    end

    return LPTResult{T}(psi1, psi2, psi3, n_order, res, grid.boxsize)
end
