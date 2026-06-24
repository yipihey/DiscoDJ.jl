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
    psi1::AbstractArray{T, 4}                       # Array on CPU, CuArray on GPU
    psi2::Union{AbstractArray{T, 4}, Nothing}
    psi3::Union{AbstractArray{T, 4}, Nothing}
    n_order::Int
    res::Int
    boxsize::T
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

# ── 1LPT ─────────────────────────────────────────────────────────────────────

function _compute_1lpt(fphi_ini::AbstractArray{Complex{T}}, grid::FourierGrid{T};
                       backend=:ka) where T
    kx, ky, kz = grid.k_vecs
    res = grid.res

    # fphi_ini is already φ₁(k) = δ(k)/k² from generate_grf (Poisson already solved)
    fphi1 = fphi_ini

    # ψ¹_d(k) = i·k_d·φ₁(k) for d = x,y,z
    psi1 = similar(fphi1, T, res, res, res, 3)   # device-aware (Array on CPU, CuArray on GPU)
    tmp  = similar(fphi1)   # complex scratch; destroyed by plan_inv each iteration
    for (d, kcomp) in enumerate((kx, ky, kz))
        _grad_multiply!(tmp, fphi1, kcomp; backend)
        psi1[:, :, :, d] .= grid.plan_inv * tmp   # plan_inv destroys tmp (c2r)
    end
    return psi1, fphi1
end

# ── 2LPT ─────────────────────────────────────────────────────────────────────
#
# The 2LPT source term is: S₂ = Σ_{i<j} (φ₁,ii φ₁,jj - φ₁,ij²)
# In Fourier space, each derivative φ₁,ij is just multiplied by i·ki·i·kj = -ki·kj.
# So the source is built from real-space products of second derivatives.

function _compute_2lpt(fphi1::AbstractArray{Complex{T}}, grid::FourierGrid{T};
                       backend=:ka) where T
    kx, ky, kz = grid.k_vecs
    k2 = grid.k2
    res = grid.res

    # One shared complex scratch buffer for all six _second_deriv calls.
    # Each call overwrites fd entirely before passing it to plan_inv, which
    # destroys the contents (c2r transform).  Net allocations: 1 instead of 6.
    fd = similar(fphi1)
    function _second_deriv(ki, kj)
        _build_second_deriv!(fd, ki, kj, fphi1)
        grid.plan_inv * fd   # fd destroyed; returns Array{T,3}
    end

    d11 = _second_deriv(kx, kx)
    d22 = _second_deriv(ky, ky)
    d33 = _second_deriv(kz, kz)
    d12 = _second_deriv(kx, ky)
    d13 = _second_deriv(kx, kz)
    d23 = _second_deriv(ky, kz)

    S2_real = @. d11*d22 - d12^2 + d11*d33 - d13^2 + d22*d33 - d23^2

    fS2   = grid.plan_fwd * S2_real   # rfft via cached plan
    fphi2 = similar(fS2)
    _inv_laplace!(fphi2, fS2, k2; backend)
    fphi2 .*= T(-3/7)

    psi2 = similar(fphi1, T, res, res, res, 3)   # device-aware
    tmp  = similar(fphi2)   # complex scratch; destroyed by plan_inv each iteration
    for (d, kcomp) in enumerate((kx, ky, kz))
        _grad_multiply!(tmp, fphi2, kcomp; backend)
        psi2[:, :, :, d] .= grid.plan_inv * tmp
    end
    return psi2, fphi2
end

# ── 3LPT ─────────────────────────────────────────────────────────────────────
#
# The 3LPT source has two parts:
#   S₃ᵃ = det(φ₁,ij)   (three-way product of first-order second derivatives)
#   S₃ᵇ = Σ φ₂,ii φ₁,jj - φ₂,ij φ₁,ji   (cross term)
# ψ³ = ∇φ₃, where ∇²φ₃ = (10/21)S₃ᵃ + S₃ᵇ/3

function _compute_3lpt(fphi1::AbstractArray{Complex{T}}, fphi2::AbstractArray{Complex{T}},
                       grid::FourierGrid{T}; backend=:ka) where T
    kx, ky, kz = grid.k_vecs
    k2 = grid.k2
    res = grid.res

    # One shared scratch buffer for all twelve _sd calls (fphi1 and fphi2 have
    # the same shape).  Each call fills fd from scratch before plan_inv destroys it.
    fd = similar(fphi1)
    function _sd(ki, kj, fphi)
        _build_second_deriv!(fd, ki, kj, fphi)
        grid.plan_inv * fd   # fd destroyed; returns Array{T,3}
    end

    d1_11 = _sd(kx, kx, fphi1); d1_22 = _sd(ky, ky, fphi1); d1_33 = _sd(kz, kz, fphi1)
    d1_12 = _sd(kx, ky, fphi1); d1_13 = _sd(kx, kz, fphi1); d1_23 = _sd(ky, kz, fphi1)

    S3a = @. d1_11*(d1_22*d1_33 - d1_23^2) -
             d1_12*(d1_12*d1_33 - d1_23*d1_13) +
             d1_13*(d1_12*d1_23 - d1_22*d1_13)

    d2_11 = _sd(kx, kx, fphi2); d2_22 = _sd(ky, ky, fphi2); d2_33 = _sd(kz, kz, fphi2)
    d2_12 = _sd(kx, ky, fphi2); d2_13 = _sd(kx, kz, fphi2); d2_23 = _sd(ky, kz, fphi2)

    S3b = @. d2_11*(d1_22 + d1_33) + d2_22*(d1_11 + d1_33) + d2_33*(d1_11 + d1_22) -
             2*(d2_12*d1_12 + d2_13*d1_13 + d2_23*d1_23)

    S3_real = @. T(10/21) * S3a + T(1/3) * S3b

    fS3   = grid.plan_fwd * S3_real   # rfft via cached plan
    fphi3 = similar(fS3)
    _inv_laplace!(fphi3, fS3, k2; backend)
    fphi3 .*= T(-1)

    psi3 = similar(fphi1, T, res, res, res, 3)   # device-aware
    tmp  = similar(fphi3)   # complex scratch; destroyed by plan_inv each iteration
    for (d, kcomp) in enumerate((kx, ky, kz))
        _grad_multiply!(tmp, fphi3, kcomp; backend)
        psi3[:, :, :, d] .= grid.plan_inv * tmp
    end
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
"""
function compute_lpt(fphi_ini::AbstractArray{Complex{T}}, grid::FourierGrid{T};
                     n_order::Int=2, backend::Symbol=:ka) where T
    n_order in (1, 2, 3) || error("n_order must be 1, 2, or 3")
    res = grid.res

    psi1, fphi1 = _compute_1lpt(fphi_ini, grid; backend)
    psi2 = nothing; fphi2 = nothing
    psi3 = nothing

    if n_order >= 2
        psi2, fphi2 = _compute_2lpt(fphi1, grid; backend)
    end
    if n_order >= 3
        psi3 = _compute_3lpt(fphi1, fphi2, grid; backend)
    end

    return LPTResult{T}(psi1, psi2, psi3, n_order, res, grid.boxsize)
end
