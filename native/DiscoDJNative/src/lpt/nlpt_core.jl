"""
Faithful port of DISCO-DJ's general-order nLPT engine (`nlpt_3d_jax.py` +
`nlpt.py`), arXiv:2010.12584.

This is a line-for-line reimplementation of the JAX `compute_core`
(general n-order, EdS growth coefficients baked in) and `compute_core_exact`
(orders 1–3, separate growth-shape fields combined later with the exact growth
factors D₂plus / D₃plusa,b,c).  It reproduces the full algorithm:

  * de-aliasing on the 3/2-rule extended grid (`pad` → `conv2_fourier` → `crop`),
  * Nyquist-zeroed spectral gradient kernels and the `-1/k²` inverse Laplacian,
  * the longitudinal source μ_L (`fmu2_sym`, `fmu2_and_C`, `fmu3`) **and** the
    transverse / curl modes C assembled into ψ via `inv_lap·(∇·μ_L − ∇×C)`.

Everything is written functionally (plain `rfft`/`irfft` + broadcasts) so the
AbstractFFTs ChainRules make it differentiable w.r.t. the initial potential.
The optimised, memory-lean `compute_lpt` (nlpt.jl) remains the forward-only path;
this module is the faithful / inference path and the parity reference.

Field layout: complex rfft arrays are `(res, res, res÷2+1)` with the half-complex
axis on dim 3, matching `rfft(x,[3,1,2])`.  Vector fields carry a trailing length-3
component axis: `(res, res, res÷2+1, 3)`.
"""

export NLPTKernels, nlpt_kernels, compute_core, compute_core_exact,
       evaluate_core, lpt_displacement, upsample_white_noise

using ChainRulesCore: @ignore_derivatives   # growth factors are constants in ω
import ChainRulesCore

# ── rfft/irfft in the pipeline's [3,1,2] convention ───────────────────────────
# CPU uses the [3,1,2] region directly (JAX-parity layout); the CUDA extension
# overrides these for CuArray with a permute-wrapped rfft (cuFFT needs an increasing
# region) — numerically identical, same logical (…,…,half-on-3) layout.
_rfftn(x) = rfft(x, [3, 1, 2])
_irfftn(f, n::Int) = irfft(f, n, [3, 1, 2])
_brfftn(f, n::Int) = brfft(f, n, [3, 1, 2])   # unnormalized inverse (adjoint building block)

# Device-aware zero allocation: `similar` follows the reference array's backend
# (Array → Array, CuArray → CuArray), so the de-aliasing `cat`s stay on-device.  The
# zero-pads are constants (no gradient flows to them), so the `fill!` is kept off the
# AD tape — otherwise Zygote errors on the in-place fill.
@inline _czeros(ref::AbstractArray, dims::Integer...) =
    @ignore_derivatives fill!(similar(ref, dims...), 0)

# ── 1-D k-vectors (DISCO-DJ get_fourier_grid convention) ──────────────────────
# Full axis: fftshift(arange(-N//2, N//2))·dk → 0,1,…,N/2-1,-N/2,…,-1 (negative
# Nyquist at index N/2).  Half axis: 0,1,…,N/2.  dk = 2π/boxsize.
function _kfull(n::Int, boxsize::T) where {T}
    dk = T(2π) / T(boxsize)
    T[(m < n ÷ 2 ? T(m) : T(m - n)) for m in 0:n-1] .* dk
end
_khalf(n::Int, boxsize::T) where {T} = collect(T, 0:n÷2) .* (T(2π) / T(boxsize))

# Spectral gradient kernel i·k along `axis` (1,2,3), reshaped & Nyquist-zeroed
# (axis 3 is the half axis → Nyquist at the last index; full axes → index N/2+1).
function _grad_kernel(n::Int, boxsize::T, axis::Int) where {T}
    if axis == 3
        ker = Complex{T}.(im .* _khalf(n, boxsize))
        ker[end] = 0
        return reshape(ker, 1, 1, :)
    else
        ker = Complex{T}.(im .* _kfull(n, boxsize))
        ker[n ÷ 2 + 1] = 0
        return axis == 1 ? reshape(ker, :, 1, 1) : reshape(ker, 1, :, 1)
    end
end

# Inverse Laplacian -1/k² (DC set so the kernel is finite; always multiplied by a
# gradient kernel which vanishes at DC, so the DC value is immaterial).
function _inv_lap_kernel(n::Int, boxsize::T) where {T}
    kx = reshape(_kfull(n, boxsize), :, 1, 1)
    ky = reshape(_kfull(n, boxsize), 1, :, 1)
    kz = reshape(_khalf(n, boxsize), 1, 1, :)
    k2 = @. kx^2 + ky^2 + kz^2
    k2[1, 1, 1] = one(T)
    return -one(T) ./ k2
end

# ── Precomputed kernels ───────────────────────────────────────────────────────
# Parametric over the array type (AC complex, AR real) so the same struct holds CPU
# `Array`s or `CuArray`s (type-stable on both) — `to_gpu(K)` just rebuilds with CuArrays.
struct NLPTKernels{T<:AbstractFloat, AC<:AbstractArray{Complex{T},3}, AR<:AbstractArray{T,3}}
    res::Int
    ext::Int          # extended (de-aliasing) resolution = 3·res÷2
    boxsize::T
    d_dx::AC; d_dy::AC; d_dz::AC
    inv_lap::AR
    dx_ext::AC; dy_ext::AC; dz_ext::AC
end

# Outer constructor: infer the array types AC, AR from the kernel arrays.
function NLPTKernels{T}(res::Int, ext::Int, bs::T, ddx::AC, ddy::AC, ddz::AC,
                        il::AR, dxe::AC, dye::AC, dze::AC) where {T, AC, AR}
    NLPTKernels{T, AC, AR}(res, ext, bs, ddx, ddy, ddz, il, dxe, dye, dze)
end

"""
    nlpt_kernels(res, boxsize; T=Float64) -> NLPTKernels

Build the base-grid spectral gradient / inverse-Laplacian kernels and the
extended-grid (3/2-rule) gradient kernels used by the de-aliasing convolutions.
"""
function nlpt_kernels(res::Int, boxsize::Real; T::Type{<:AbstractFloat}=Float64)
    bs  = T(boxsize)
    ext = 3 * res ÷ 2
    NLPTKernels{T}(res, ext, bs,
        _grad_kernel(res, bs, 1), _grad_kernel(res, bs, 2), _grad_kernel(res, bs, 3),
        _inv_lap_kernel(res, bs),
        _grad_kernel(ext, bs, 1), _grad_kernel(ext, bs, 2), _grad_kernel(ext, bs, 3))
end

# ── De-aliasing primitives (pad / crop / conv2_fourier) ───────────────────────
# Written functionally with `cat` (no `setindex!`) so the AbstractFFTs/Zygote
# ChainRules differentiate straight through.  Each dimension is split into its
# lower / (Nyquist-dropping) / upper blocks and reassembled at the other size.
#
# Note: `pad` is always immediately multiplied by an extended gradient kernel
# (`_dext`), which vanishes at DC — so the explicit DC-zeroing in the JAX `pad`
# is redundant here and omitted (parity is unaffected, verified numerically).

"""Pad an rfft field `(orig,orig,orig÷2+1)` into the extended `(ext,ext,ext÷2+1)`
layout (high modes → 0), with the (ext/orig)³ amplitude rescale."""
function _pad3(ff::AbstractArray{Complex{T},3}, orig::Int, ext::Int) where {T}
    h = orig ÷ 2
    # dim 1 (full): [lo | zeros | hi]
    a1 = cat(ff[1:h, :, :], _czeros(ff, ext - 2h + 1, orig, h + 1),
             ff[h+2:orig, :, :]; dims=1)                              # (ext, orig, h+1)
    # dim 2 (full)
    a2 = cat(a1[:, 1:h, :], _czeros(ff, ext, ext - 2h + 1, h + 1),
             a1[:, h+2:orig, :]; dims=2)                              # (ext, ext, h+1)
    # dim 3 (half): only the lower block, zero-pad up to ext÷2+1
    a3 = cat(a2[:, :, 1:h], _czeros(ff, ext, ext, ext ÷ 2 + 1 - h); dims=3)
    return a3 .* T((ext / orig)^3)
end

"""Crop an extended rfft field back to `(orig,orig,orig÷2+1)` (inverse block map of
`_pad3`, Nyquist planes → 0), with the (orig/ext)³ amplitude rescale."""
function _crop3(ff::AbstractArray{Complex{T},3}, orig::Int, ext::Int) where {T}
    h = orig ÷ 2
    nz = size(ff, 3)
    # dim 1 (full): [lo | Nyquist=0 | hi]
    a1 = cat(ff[1:h, :, :], _czeros(ff, 1, ext, nz), ff[ext-h+2:ext, :, :]; dims=1)   # (orig, ext, nz)
    # dim 2 (full)
    a2 = cat(a1[:, 1:h, :], _czeros(ff, orig, 1, nz), a1[:, ext-h+2:ext, :]; dims=2)  # (orig, orig, nz)
    # dim 3 (half): lower block + zero Nyquist plane
    a3 = cat(a2[:, :, 1:h], _czeros(ff, orig, orig, orig ÷ 2 + 1 - h); dims=3)
    return a3 .* T((orig / ext)^3)
end

"""
    upsample_white_noise(ω, res_fine) -> ω_fine

Coarse→fine warm-start upsample of a real white-noise field `ω` (res,res,res), by Fourier
zero-padding the low-k modes with the `(res/res_fine)^{3/2}` per-mode rescale — the MUSIC
construction.  The shared low-k modes are preserved so the IC potential
`white_noise_to_fphi` (∝ √((res/L)³)·rfft(ω)/k²) matches the coarse realisation at large
scales (corr & amplitude ≈ 1 down to the coarse Nyquist); the new high-k modes are zero, to
be filled in by optimisation at the finer level.  Differentiable (plain rfft/pad/irfft).
"""
function upsample_white_noise(ω::AbstractArray{T,3}, res_fine::Int) where {T}
    res = size(ω, 1)
    res_fine == res && return copy(ω)
    res_fine > res || throw(ArgumentError("upsample_white_noise: res_fine ($res_fine) must exceed res ($res)"))
    return _irfftn(_pad3(_rfftn(ω), res, res_fine) .* T((res / res_fine)^(T(3) / 2)), res_fine)
end

"""Fourier-space convolution: rfft(irfft(ff1)·irfft(ff2)); inputs are extended
layout, output cropped to base resolution unless `do_crop=false`."""
function _conv2(ff1::AbstractArray{Complex{T},3}, ff2::AbstractArray{Complex{T},3},
                orig::Int, ext::Int; do_crop::Bool=true) where {T}
    prod = _irfftn(ff1, ext) .* _irfftn(ff2, ext)
    f = _rfftn(prod)
    return do_crop ? _crop3(f, orig, ext) : f
end

# Extended derivative of vector-field component m (1..3) along axis (1..3):
#   derivs_ext[axis] · pad(field[:,:,:,m])
@inline function _dext(K::NLPTKernels{T}, X::AbstractArray{Complex{T},4}, m::Int, axis::Int) where {T}
    d = axis == 1 ? K.dx_ext : axis == 2 ? K.dy_ext : K.dz_ext
    return d .* _pad3(X[:, :, :, m], K.res, K.ext)
end

# ── μ₂ symmetric term (j == i-j), Algorithm 1 ─────────────────────────────────
# fmu2_sym(f) = Σ s · conv2(∂_k f_i, ∂_l f_j) over the six (i,j,k,l,s) below.
const _MU2_SYM_TERMS = ((1,2,1,2,1), (1,3,1,3,1), (2,3,2,3,1),
                        (1,2,2,1,-1), (1,3,3,1,-1), (2,3,3,2,-1))

function fmu2_sym(K::NLPTKernels{T}, f1::AbstractArray{Complex{T},4}) where {T}
    res, ext = K.res, K.ext
    acc = _czeros(f1, res, res, res ÷ 2 + 1)
    for (i, j, k, l, s) in _MU2_SYM_TERMS
        t1 = _dext(K, f1, i, k)
        t2 = _dext(K, f1, j, l)
        acc = acc .+ T(s) .* _conv2(t1, t2, res, ext)
    end
    return acc
end



# ── μ₂ & C asymmetric terms (j != i-j), Algorithms 1 & 3 ──────────────────────
# 1-based ports of the JAX i/j/k/l/s lists (component indices i,j and axes k,l).
const _MU2_I = (1,1,2,3,2,2,3,1,3,3,1,2)
const _MU2_J = (2,3,1,1,3,1,2,2,1,2,3,3)
const _MU2_K = (1,1,1,1,2,2,2,2,3,3,3,3)
const _MU2_L = (2,3,2,3,3,1,3,1,1,2,1,2)
const _MU2_S = (1,1,-1,-1,1,1,-1,-1,1,1,-1,-1)
# C base lists (Cx); Cy, Cz are these with component+axis indices cycled by 1, 2.
const _C_I = (1,1,2,2,3,3); const _C_J = (1,1,2,2,3,3)
const _C_K = (2,3,2,3,2,3); const _C_L = (3,2,3,2,3,2)
const _C_S = (1,-1,1,-1,1,-1)
@inline _cyc(x::Int, n::Int) = (x - 1 + n) % 3 + 1   # cycle component/axis by n (1-based)

function fmu2_and_C(K::NLPTKernels{T}, f1::AbstractArray{Complex{T},4},
                    f2::AbstractArray{Complex{T},4}) where {T}
    res, ext = K.res, K.ext
    z() = _czeros(f1, res, res, res ÷ 2 + 1)
    mu2 = z()
    for n in 1:12
        t1 = _dext(K, f1, _MU2_I[n], _MU2_K[n])
        t2 = _dext(K, f2, _MU2_J[n], _MU2_L[n])
        mu2 = mu2 .+ T(_MU2_S[n]) .* _conv2(t1, t2, res, ext)
    end
    Ccomp(shift) = begin
        acc = z()
        for n in 1:6
            i = _cyc(_C_I[n], shift); j = _cyc(_C_J[n], shift)
            k = _cyc(_C_K[n], shift); l = _cyc(_C_L[n], shift)
            t1 = _dext(K, f1, i, k); t2 = _dext(K, f2, j, l)
            acc = acc .+ T(_C_S[n]) .* _conv2(t1, t2, res, ext)
        end
        acc
    end
    C = cat(Ccomp(0), Ccomp(1), Ccomp(2); dims=4)
    return mu2, C
end

# ── μ₃ term, Algorithm 2 ──────────────────────────────────────────────────────
const _MU3_K = (1,1,2,2,3,3); const _MU3_L = (2,3,3,1,1,2)
const _MU3_M = (3,2,1,3,2,1); const _MU3_S = (1,-1,1,-1,1,-1)

function fmu3(K::NLPTKernels{T}, f1::AbstractArray{Complex{T},4},
              f2::AbstractArray{Complex{T},4}, f3::AbstractArray{Complex{T},4}) where {T}
    res, ext = K.res, K.ext
    acc = _czeros(f1, res, res, res ÷ 2 + 1)
    for n in 1:6
        termA = _dext(K, f1, 1, _MU3_K[n])                       # ∂_k A_1
        inner = _conv2(_dext(K, f2, 2, _MU3_L[n]),               # ∂_l B_2
                       _dext(K, f3, 3, _MU3_M[n]), res, ext; do_crop=false)  # ∂_m C_3
        acc = acc .+ T(_MU3_S[n]) .* _conv2(termA, inner, res, ext)
    end
    return acc
end

# ── ψ assembly: longitudinal ∇·μ_L minus transverse curl ∇×C ──────────────────
# ψ_x = inv_lap·(∂x·fL − (∂y·C_z − ∂z·C_y)), and cyclically.
function _assemble_psi(K::NLPTKernels{T}, fL::AbstractArray{Complex{T},3},
                       fT::Union{Nothing,AbstractArray{Complex{T},4}}) where {T}
    dx, dy, dz, il = K.d_dx, K.d_dy, K.d_dz, K.inv_lap
    if fT === nothing
        px = il .* (dx .* fL); py = il .* (dy .* fL); pz = il .* (dz .* fL)
    else
        Cx = @view fT[:, :, :, 1]; Cy = @view fT[:, :, :, 2]; Cz = @view fT[:, :, :, 3]
        px = il .* (dx .* fL .- (dy .* Cz .- dz .* Cy))
        py = il .* (dy .* fL .- (dz .* Cx .- dx .* Cz))
        pz = il .* (dz .* fL .- (dx .* Cy .- dy .* Cx))
    end
    return cat(px, py, pz; dims=4)
end

# Pure-transverse ψ (no longitudinal part) — used for psi_3c_ex.
function _assemble_psi_transverse(K::NLPTKernels{T}, fT::AbstractArray{Complex{T},4}) where {T}
    dx, dy, dz, il = K.d_dx, K.d_dy, K.d_dz, K.inv_lap
    Cx = @view fT[:, :, :, 1]; Cy = @view fT[:, :, :, 2]; Cz = @view fT[:, :, :, 3]
    px = il .* (-(dy .* Cz .- dz .* Cy))
    py = il .* (-(dz .* Cx .- dx .* Cz))
    pz = il .* (-(dx .* Cy .- dy .* Cx))
    return cat(px, py, pz; dims=4)
end

# Zel'dovich displacement in Fourier space: ψ₁ = -[∂x,∂y,∂z] φ (component axis 4).
function _psi1_fourier(K::NLPTKernels{T}, fphi::AbstractArray{Complex{T},3}) where {T}
    cat(-(K.d_dx .* fphi), -(K.d_dy .* fphi), -(K.d_dz .* fphi); dims=4)
end

# irfft each of the 3 components of a Fourier vector field → real (res,res,res,3).
# Unrolled (no generator/view) so Zygote traces it cleanly.
function _vec_irfftn(K::NLPTKernels{T}, V::AbstractArray{Complex{T},4}) where {T}
    res = K.res
    c1 = reshape(_irfftn(V[:, :, :, 1], res), res, res, res, 1)
    c2 = reshape(_irfftn(V[:, :, :, 2], res), res, res, res, 1)
    c3 = reshape(_irfftn(V[:, :, :, 3], res), res, res, res, 1)
    return cat(c1, c2, c3; dims=4)
end

# ── General-order compute_core (EdS growth coefficients) ──────────────────────
"""
    compute_core(fphi_ini, K; n_order) -> Dict("psi_1"=>…, "psi_2"=>…, …)

General n-order nLPT with EdS growth coefficients baked into each order (the JAX
`compute_core`).  Returns the real-space displacement shape fields ψ_n; combine
with `D₁(a)ⁿ` (see `evaluate`).  Longitudinal + transverse modes, de-aliased.
"""
function compute_core(fphi_ini::AbstractArray{Complex{T},3}, K::NLPTKernels{T};
                      n_order::Int=2, no_transverse::Bool=false) where {T}
    res, ext = K.res, K.ext
    # JAX zeros φ's DC mode here; redundant — every use of φ multiplies by a
    # gradient kernel (DC = 0), so we pass fphi_ini through untouched.
    # ψ_i in Fourier, accumulated in a tuple (no in-place `push!` → Zygote-traceable)
    psi = (_psi1_fourier(K, fphi_ini),)

    for i in 2:n_order
        fL = _czeros(fphi_ini, res, res, res ÷ 2 + 1)
        fT = _czeros(fphi_ini, res, res, res ÷ 2 + 1, 3)

        # symmetric μ₂ term for even orders (j == i/2)
        if iseven(i)
            h = i ÷ 2
            fac_sym = T(((3 - i) / 2 - h^2 - h^2) / ((i + 3 / 2) * (i - 1)))
            fL = fL .+ fac_sym .* fmu2_sym(K, psi[h])
        end

        if i > 2
            # asymmetric μ₂ & C over j = 1 .. (i+1)÷2-1, i-j = i-1 .. i÷2+1
            for j in 1:((i + 1) ÷ 2 - 1)
                imj = i - j
                fac_mu2 = T(((3 - i) / 2 - j^2 - imj^2) / ((i + 3 / 2) * (i - 1)))
                fac_C   = T(1 - 2j / i)
                mu2, C = fmu2_and_C(K, psi[j], psi[imj])
                fL = fL .+ fac_mu2 .* mu2
                if !no_transverse
                    fT = fT .+ fac_C .* C
                end
            end
            # μ₃ over k = 1..i-2, l = 1..i-k-1
            for k in 1:(i - 2)
                for l in 1:(i - k - 1)
                    m = i - k - l
                    fac = T(((3 - i) / 2 - k^2 - l^2 - m^2) / ((i + 3 / 2) * (i - 1)))
                    fL = fL .+ fac .* fmu3(K, psi[k], psi[l], psi[m])
                end
            end
        end

        psi = (psi..., _assemble_psi(K, fL, no_transverse ? nothing : fT))
    end

    out = Dict{String,AbstractArray{T,4}}()
    for i in 1:n_order
        out["psi_$i"] = _vec_irfftn(K, psi[i])
    end
    return out
end

# ── compute_core_exact (orders 1–3, separate growth-shape fields) ─────────────
"""
    compute_core_exact(fphi_ini, K; n_order) -> Dict

Exact-growth nLPT (JAX `compute_core_exact`): returns the growth-factor-free shape
fields `psi_1`, `psi_2_ex`, `psi_3a_ex`, `psi_3b_ex`, `psi_3c_ex` in real space.
Combine with D₁, D₂plus, D₃plusa/b/c (see `evaluate`).
"""
function compute_core_exact(fphi_ini::AbstractArray{Complex{T},3}, K::NLPTKernels{T};
                            n_order::Int=3) where {T}
    res, ext = K.res, K.ext
    psi1 = _psi1_fourier(K, fphi_ini)   # φ DC irrelevant (killed by gradient kernel)

    out = Dict{String,AbstractArray{T,4}}()
    out["psi_1"] = _vec_irfftn(K, psi1)

    if n_order >= 2
        fL2 = fmu2_sym(K, psi1)
        psi2 = _assemble_psi(K, fL2, nothing)        # longitudinal only
        out["psi_2_ex"] = _vec_irfftn(K, psi2)

        if n_order >= 3
            # 3a: longitudinal, from μ₃(ψ₁,ψ₁,ψ₁) with factor -1
            fL3a = (-one(T)) .* fmu3(K, psi1, psi1, psi1)
            psi3a = _assemble_psi(K, fL3a, nothing)
            # 3b (longitudinal) & 3c (transverse), from μ₂&C(ψ₁, ψ₂_ex)
            mu2, C = fmu2_and_C(K, psi1, psi2)
            fL3b = (-T(0.5)) .* mu2
            fT3c = (-one(T)) .* C
            psi3b = _assemble_psi(K, fL3b, nothing)
            psi3c = _assemble_psi_transverse(K, fT3c)
            out["psi_3a_ex"] = _vec_irfftn(K, psi3a)
            out["psi_3b_ex"] = _vec_irfftn(K, psi3b)
            out["psi_3c_ex"] = _vec_irfftn(K, psi3c)
        end
    end
    return out
end

# ── Growth-factor combination ψ(a) = Σ Dₙ(a)·ψ_n (matches JAX NLPT.evaluate) ───
"""
    evaluate_core(shapes, cosmo, a; n_order, exact_growth) -> Array (res,res,res,3)

Combine the nLPT shape fields with the growth factors, exactly as DISCO-DJ's
`NLPT.evaluate`:

  * `exact_growth=true`  → D₁·ψ₁ + D₂plus·ψ₂ₑₓ + D₃plusa·ψ₃ₐₑₓ + D₃plusb·ψ₃ᵦₑₓ
                           + D₃plusc·ψ₃ᵧₑₓ  (shapes from `compute_core_exact`)
  * `exact_growth=false` → Σₙ D₁(a)ⁿ · ψ_n  (shapes from `compute_core`)
"""
function evaluate_core(shapes::Dict{String,AbstractArray{T,4}}, cosmo::Cosmology, a::Real;
                       n_order::Int, exact_growth::Bool) where {T}
    # Growth factors depend only on (cosmo, a), not on ω/φ — keep them off the AD
    # tape (Zygote tracing the growth-table interpolation otherwise segfaults).
    if exact_growth
        D1  = @ignore_derivatives T(growth_D1(cosmo, a))
        psi = D1 .* shapes["psi_1"]
        if n_order >= 2
            D2 = @ignore_derivatives T(growth_D2(cosmo, a))
            psi = psi .+ D2 .* shapes["psi_2_ex"]
        end
        if n_order >= 3
            D3a = @ignore_derivatives T(growth_D3a(cosmo, a))
            D3b = @ignore_derivatives T(growth_D3b(cosmo, a))
            D3c = @ignore_derivatives T(growth_D3c(cosmo, a))
            psi = psi .+ D3a .* shapes["psi_3a_ex"] .+
                         D3b .* shapes["psi_3b_ex"] .+
                         D3c .* shapes["psi_3c_ex"]
        end
        return psi
    else
        D1  = @ignore_derivatives T(growth_D1(cosmo, a))
        psi = D1 .* shapes["psi_1"]
        for n in 2:n_order
            Dn = @ignore_derivatives D1^n
            psi = psi .+ Dn .* shapes["psi_$n"]
        end
        return psi
    end
end

"""
    lpt_displacement(fphi, K, cosmo, a; n_order=2, exact_growth=false) -> (res,res,res,3)

Faithful one-shot nLPT displacement ψ(a): compute the shape fields and combine
them with the growth factors.  `K = nlpt_kernels(res, boxsize)`.
"""
function lpt_displacement(fphi::AbstractArray{Complex{T},3}, K::NLPTKernels{T},
                          cosmo::Cosmology, a::Real; n_order::Int=2,
                          exact_growth::Bool=false) where {T}
    shapes = exact_growth ? compute_core_exact(fphi, K; n_order) :
                            compute_core(fphi, K; n_order)
    return evaluate_core(shapes, cosmo, a; n_order, exact_growth)
end

# ── Analytic adjoint for the de-aliased 2LPT bilinear (μ₂ is QUADRATIC in Ψ₁ ⇒ vjp = ONE linear map) ──
# μ₂ is a symmetric quadratic form in Ψ₁.  crop∘rfft is linear, so the six de-aliased products collapse to
#   μ₂ = crop( rfft( Σ s·Aᵢₖ·A_jl ) ),   Aᵢₖ = irfft(∂ₖΨ₁ᵢ)   (the NINE shared real 2nd-derivative fields),
# computed ONCE.  The vjp is then a single explicit pass — P̄ = rfft†(crop†(ā)); each field's cotangent
# Āᵢₖ = P̄·(its partner from the six terms); back through the dext/irfft adjoints — NO Zygote, NO re-diff,
# NO tape.  ~10 FFTs each way vs the term-by-term streaming rrule's re-diff (≈24 GB pool → ≈2 GB at res 192).
# The Aᵢₖ, dext†, crop†/pad† and lazy-weight rfft†/irfft† pieces are the FD-validated hand rrules above.
function ChainRulesCore.rrule(::typeof(fmu2_sym), K::NLPTKernels{T},
                              f1::AbstractArray{Complex{T},4}) where {T}
    res, ext = K.res, K.ext
    A = ntuple(i -> ntuple(k -> _irfftn(_dext(K, f1, i, k), ext), 3), 3)   # A[i][k], ext real (shared)
    μreal = A[1][1] .* A[2][2] .+ A[1][1] .* A[3][3] .+ A[2][2] .* A[3][3] .-
            A[1][2] .* A[2][1] .- A[1][3] .* A[3][1] .- A[2][3] .* A[3][2]
    y  = _crop3(_rfftn(μreal), res, ext)
    nh = ext ÷ 2 + 1; N = T(ext)^3
    wv = _halfdim_weights(f1, nh, ext, T)
    ker(k) = k == 1 ? K.dx_ext : k == 2 ? K.dy_ext : K.dz_ext
    # ∂μreal/∂Aᵢₖ (the "partner" of field (i,k) summed over the six terms)
    partner(i, k) =
        (i, k) == (1, 1) ? (A[2][2] .+ A[3][3]) : (i, k) == (2, 2) ? (A[1][1] .+ A[3][3]) :
        (i, k) == (3, 3) ? (A[1][1] .+ A[2][2]) : (i, k) == (1, 2) ? (.-A[2][1]) :
        (i, k) == (2, 1) ? (.-A[1][2]) : (i, k) == (1, 3) ? (.-A[3][1]) :
        (i, k) == (3, 1) ? (.-A[1][3]) : (i, k) == (2, 3) ? (.-A[3][2]) : (.-A[2][3])
    function fmu2_sym_pullback(ā)
        ā_ = ChainRulesCore.unthunk(ā)
        P̄  = _brfftn((_pad3(ā_, res, ext) .* T((res / ext)^6)) ./ wv, ext)   # crop† then rfft† → μreal cotangent
        f̄  = fill!(similar(f1), zero(Complex{T}))
        for i in 1:3, k in 1:3
            Ā    = P̄ .* partner(i, k)                                        # Aᵢₖ cotangent
            dcot = _rfftn(Ā) .* (wv ./ N)                                     # irfft† → ∂ₖΨ₁ᵢ cotangent
            s̄    = _crop3(conj.(ker(k)) .* dcot, res, ext) .* T((ext / res)^6) # dext† (crop side)
            @views f̄[:, :, :, i] .+= s̄
        end
        return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), f̄)
    end
    return y, fmu2_sym_pullback
end

# ── Shared adjoint pieces for the analytic LPT bilinears (all via the FD-validated hand adjoints) ────
_Aext(K::NLPTKernels, f, i::Int, k::Int) = _irfftn(_dext(K, f, i, k), K.ext)   # ∂ₖ fᵢ  (ext real)
# (crop∘rfft)† : res-rfft cotangent ā → ext-real cotangent of the pre-crop real product
_cr_adj(K::NLPTKernels{T}, ā, wv) where {T} =
    _brfftn((_pad3(ChainRulesCore.unthunk(ā), K.res, K.ext) .* T((K.res / K.ext)^6)) ./ wv, K.ext)
# accumulate dext†(irfft†(Ā)) for field (component i, axis k) into f̄[:,:,:,i]
function _dext_accum!(f̄, K::NLPTKernels{T}, Ā, i::Int, k::Int, wv, N) where {T}
    dk = k == 1 ? K.dx_ext : k == 2 ? K.dy_ext : K.dz_ext
    @views f̄[:, :, :, i] .+= _crop3(conj.(dk) .* (_rfftn(Ā) .* (wv ./ N)), K.res, K.ext) .* T((K.ext / K.res)^6)
    return f̄
end

# ── Analytic adjoint for the μ₃ trilinear (the 3LPT source) ─────────────────────────────────────────
# The nested `_conv2(…, do_crop=false)` is a de-aliased TRIPLE product (irfft∘rfft round-trips on the ext
# grid), so  fmu3 = crop( rfft( Σ s·A1_k·B_l·Cc_m ) )  — the Levi-Civita triple product of the nine ext-real
# fields A1_a=irfft(∂ₐf1₁), B_a=irfft(∂ₐf2₂), Cc_a=irfft(∂ₐf3₃), computed ONCE.  Trilinear ⇒ each arg's vjp
# is a quadratic map; ChainRules sums the three when called as fmu3(ψ₁,ψ₁,ψ₁).  No Zygote, no tape.
function ChainRulesCore.rrule(::typeof(fmu3), K::NLPTKernels{T}, f1::AbstractArray{Complex{T},4},
                              f2::AbstractArray{Complex{T},4}, f3::AbstractArray{Complex{T},4}) where {T}
    ext = K.ext
    A1 = ntuple(a -> _Aext(K, f1, 1, a), 3); B = ntuple(a -> _Aext(K, f2, 2, a), 3); Cc = ntuple(a -> _Aext(K, f3, 3, a), 3)
    μ = A1[1] .* B[2] .* Cc[3] .- A1[1] .* B[3] .* Cc[2] .+ A1[2] .* B[3] .* Cc[1] .-
        A1[2] .* B[1] .* Cc[3] .+ A1[3] .* B[1] .* Cc[2] .- A1[3] .* B[2] .* Cc[1]
    y  = _crop3(_rfftn(μ), K.res, ext)
    nh = ext ÷ 2 + 1; N = T(ext)^3; wv = _halfdim_weights(f1, nh, ext, T)
    function fmu3_pullback(ā)
        P̄  = _cr_adj(K, ā, wv)
        f̄1 = fill!(similar(f1), zero(Complex{T})); f̄2 = fill!(similar(f2), zero(Complex{T})); f̄3 = fill!(similar(f3), zero(Complex{T}))
        dA1 = (B[2].*Cc[3].-B[3].*Cc[2], B[3].*Cc[1].-B[1].*Cc[3], B[1].*Cc[2].-B[2].*Cc[1])
        dB  = (A1[3].*Cc[2].-A1[2].*Cc[3], A1[1].*Cc[3].-A1[3].*Cc[1], A1[2].*Cc[1].-A1[1].*Cc[2])
        dCc = (A1[2].*B[3].-A1[3].*B[2], A1[3].*B[1].-A1[1].*B[3], A1[1].*B[2].-A1[2].*B[1])
        for a in 1:3
            _dext_accum!(f̄1, K, P̄ .* dA1[a], 1, a, wv, N)
            _dext_accum!(f̄2, K, P̄ .* dB[a],  2, a, wv, N)
            _dext_accum!(f̄3, K, P̄ .* dCc[a], 3, a, wv, N)
        end
        return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), f̄1, f̄2, f̄3)
    end
    return y, fmu3_pullback
end

# ── Analytic adjoint for the μ₂&C bilinear (asymmetric 3LPT source + transverse curl) ────────────────
# Bilinear in (f1,f2) with two outputs: μ₂ = crop(rfft(Σ_{12} s·A1ᵢₖ·A2_jl)) and C_c = crop(rfft(Σ_6 s·…)),
# with A1ᵢₖ=irfft(∂ₖf1ᵢ), A2_jl=irfft(∂ₗf2ⱼ) shared across all terms.  The vjp accumulates the per-field
# cotangents Ā1ᵢₖ, Ā2_jl over the 12 μ₂ + 18 C terms (each = ±P̄·partner), then back through dext†/irfft†.
function ChainRulesCore.rrule(::typeof(fmu2_and_C), K::NLPTKernels{T},
                              f1::AbstractArray{Complex{T},4}, f2::AbstractArray{Complex{T},4}) where {T}
    res, ext = K.res, K.ext
    A1 = [_Aext(K, f1, i, k) for i in 1:3, k in 1:3]      # (3,3) ext real
    A2 = [_Aext(K, f2, j, l) for j in 1:3, l in 1:3]
    mu2r = T(_MU2_S[1]) .* A1[_MU2_I[1], _MU2_K[1]] .* A2[_MU2_J[1], _MU2_L[1]]
    for n in 2:12
        mu2r = mu2r .+ T(_MU2_S[n]) .* A1[_MU2_I[n], _MU2_K[n]] .* A2[_MU2_J[n], _MU2_L[n]]
    end
    Creal = ntuple(c -> begin
        acc = T(_C_S[1]) .* A1[_cyc(_C_I[1], c-1), _cyc(_C_K[1], c-1)] .* A2[_cyc(_C_J[1], c-1), _cyc(_C_L[1], c-1)]
        for n in 2:6
            acc = acc .+ T(_C_S[n]) .* A1[_cyc(_C_I[n], c-1), _cyc(_C_K[n], c-1)] .* A2[_cyc(_C_J[n], c-1), _cyc(_C_L[n], c-1)]
        end
        acc
    end, 3)
    mu2 = _crop3(_rfftn(mu2r), res, ext)
    C   = cat(_crop3(_rfftn(Creal[1]), res, ext), _crop3(_rfftn(Creal[2]), res, ext),
              _crop3(_rfftn(Creal[3]), res, ext); dims=4)
    nh = ext ÷ 2 + 1; N = T(ext)^3; wv = _halfdim_weights(f1, nh, ext, T)
    function fmu2C_pullback(ā)
        ā_ = ChainRulesCore.unthunk(ā)
        m̄2 = ChainRulesCore.unthunk(ā_[1]); C̄ = ChainRulesCore.unthunk(ā_[2])
        Pm = m̄2 isa ChainRulesCore.AbstractZero ? nothing : _cr_adj(K, m̄2, wv)
        PC = ntuple(c -> C̄ isa ChainRulesCore.AbstractZero ? nothing : _cr_adj(K, C̄[:, :, :, c], wv), 3)
        Ā1 = [fill!(similar(A1[1, 1]), zero(T)) for _ in 1:3, _ in 1:3]
        Ā2 = [fill!(similar(A2[1, 1]), zero(T)) for _ in 1:3, _ in 1:3]
        if Pm !== nothing
            for n in 1:12
                i, k, j, l, s = _MU2_I[n], _MU2_K[n], _MU2_J[n], _MU2_L[n], T(_MU2_S[n])
                Ā1[i, k] .+= s .* Pm .* A2[j, l]; Ā2[j, l] .+= s .* Pm .* A1[i, k]
            end
        end
        for c in 1:3
            PC[c] === nothing && continue
            for n in 1:6
                i = _cyc(_C_I[n], c-1); k = _cyc(_C_K[n], c-1); j = _cyc(_C_J[n], c-1); l = _cyc(_C_L[n], c-1); s = T(_C_S[n])
                Ā1[i, k] .+= s .* PC[c] .* A2[j, l]; Ā2[j, l] .+= s .* PC[c] .* A1[i, k]
            end
        end
        f̄1 = fill!(similar(f1), zero(Complex{T})); f̄2 = fill!(similar(f2), zero(Complex{T}))
        for i in 1:3, k in 1:3
            _dext_accum!(f̄1, K, Ā1[i, k], i, k, wv, N)
            _dext_accum!(f̄2, K, Ā2[i, k], i, k, wv, N)
        end
        return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), f̄1, f̄2)
    end
    return (mu2, C), fmu2C_pullback
end

# Linear hand adjoint for ψ₁(k) = −∇φ: avoids taping the three kernel-broadcast operands.
function ChainRulesCore.rrule(::typeof(_psi1_fourier), K::NLPTKernels{T},
                              fphi::AbstractArray{Complex{T},3}) where {T}
    y = _psi1_fourier(K, fphi)
    function psi1_pullback(ȳ)
        Ȳ = ChainRulesCore.unthunk(ȳ)
        f̄ = .-(conj.(K.d_dx) .* Ȳ[:,:,:,1] .+ conj.(K.d_dy) .* Ȳ[:,:,:,2] .+ conj.(K.d_dz) .* Ȳ[:,:,:,3])
        return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), f̄)
    end
    return y, psi1_pullback
end

# ── Linear hand adjoints for the ext-grid block maps (kill the pad/crop cat-chain tape) ────────────
# `_pad3(x) = (ext/orig)³·S(x)` and `_crop3(y) = (orig/ext)³·Sᵀ(y)` share the same block-selection S
# (Nyquist planes dropped/zeroed), so each is the other's adjoint up to the amplitude factors:
#   pad3ᵀ(ȳ)  = (ext/orig)³·Sᵀ(ȳ) = (ext/orig)⁶·_crop3(ȳ)
#   crop3ᵀ(x̄) = (orig/ext)³·S(x̄)  = (orig/ext)⁶·_pad3(x̄)
# Verified against finite differences through the full 2LPT loss (F64 rel ~6e-6).
function ChainRulesCore.rrule(::typeof(_pad3), ff::AbstractArray{Complex{T},3},
                              orig::Int, ext::Int) where {T}
    y = _pad3(ff, orig, ext)
    function pad3_pullback(ȳ)
        Ȳ = ChainRulesCore.unthunk(ȳ)
        f̄ = _crop3(Ȳ, orig, ext) .* T((ext / orig)^6)
        return (ChainRulesCore.NoTangent(), f̄, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent())
    end
    return y, pad3_pullback
end

function ChainRulesCore.rrule(::typeof(_crop3), ff::AbstractArray{Complex{T},3},
                              orig::Int, ext::Int) where {T}
    y = _crop3(ff, orig, ext)
    function crop3_pullback(ȳ)
        Ȳ = ChainRulesCore.unthunk(ȳ)
        f̄ = _pad3(Ȳ, orig, ext) .* T((orig / ext)^6)
        return (ChainRulesCore.NoTangent(), f̄, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent())
    end
    return y, crop3_pullback
end

# `_dext(K, X, m, axis) = d_axis .* _pad3(X[:,:,:,m])` — linear; the pullback needs only the (static)
# kernel and the pad adjoint, so nothing from the forward is taped.
function ChainRulesCore.rrule(::typeof(_dext), K::NLPTKernels{T},
                              X::AbstractArray{Complex{T},4}, m::Int, axis::Int) where {T}
    y = _dext(K, X, m, axis)
    res, ext = K.res, K.ext
    function dext_pullback(ȳ)
        Ȳ = ChainRulesCore.unthunk(ȳ)
        d = axis == 1 ? K.dx_ext : axis == 2 ? K.dy_ext : K.dz_ext
        s̄ = _crop3(conj.(d) .* Ȳ, res, ext) .* T((ext / res)^6)
        X̄ = ChainRulesCore.@thunk begin
            full = fill!(similar(X), zero(Complex{T}))
            copyto!(view(full, :, :, :, m), s̄)
            full
        end
        return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), X̄,
                ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent())
    end
    return y, dext_pullback
end

# ── Staged 2LPT pipeline (fully analytic — no checkpointing needed) ────────────────────────────────
# Every piece here is analytic with a hand rrule: `fmu2_sym` (the quadratic bilinear, above),
# `_assemble_psi` (a linear Fourier map), and `_vec_irfftn` (per-component `_irfftn` hand adjoints).
# So plain Zygote composition backprops through the hand rrules with NO re-differentiation and NO
# segment tape — the analytic `fmu2_sym` closure holds only its 9 shared real fields.  (Previously
# these two stages carried `rrule_via_ad` checkpoints, which existed solely to bound the *taped*
# fmu2_sym; the analytic adjoint makes them pure overhead, so they are removed.)
export psi2_fourier, shape_stack_2lpt

"""ψ₂(k) from ψ₁(k): the de-aliased 2LPT source assembled to a displacement (longitudinal)."""
psi2_fourier(K::NLPTKernels{T}, psi1::AbstractArray{Complex{T},4}) where {T} =
    _assemble_psi(K, fmu2_sym(K, psi1), nothing)

"""(ψ₁,ψ₂)(k) → the real exact-growth shape stack (N,3,2) (2LPT layout of `exact_shape_stack`)."""
function shape_stack_2lpt(K::NLPTKernels{T}, psi1::AbstractArray{Complex{T},4},
                          psi2::AbstractArray{Complex{T},4}) where {T}
    res = K.res; N = res^3
    r1 = _vec_irfftn(K, psi1); r2 = _vec_irfftn(K, psi2)
    return cat(reshape(r1, N, 3), reshape(r2, N, 3); dims=3)
end

# ── Hand FFT adjoints with lazy weights (replace closure-captured full-size scale arrays) ──────────
# AbstractFFTs' irfft/rfft rrules capture a FULL-SIZE scale array in every pullback closure — at 512³
# several live closures cost ~8–10 GB. The adjoints only need a per-mode weight along the halved dim
# (2 for interior kz, 1 for kz=0/Nyquist) and the 1/N normalization, carried here as a (1,1,nh) vector:
#   irfft†:  X̄ = rfft(ȳ) .* (w/N)          rfft†:  x̄ = brfft(Ȳ .* (1/w))
# Verified against finite differences through the full 2LPT loss (F64 rel ~6e-6).
function _halfdim_weights(ref::AbstractArray, nh::Int, d::Int, ::Type{T}) where {T}
    wh = ones(T, nh); for k in 2:nh; wh[k] = T(2); end
    if iseven(d); wh[nh] = T(1); end
    wv = @ignore_derivatives (y = similar(ref, T, 1, 1, nh); copyto!(y, reshape(wh, 1, 1, nh)); y)
    return wv
end

function ChainRulesCore.rrule(::typeof(_irfftn), f::AbstractArray{Complex{T},3}, d::Int) where {T}
    y = _irfftn(f, d)
    n1, n2 = size(f, 1), size(f, 2); nh = size(f, 3); N = T(n1) * T(n2) * T(d)
    function irfftn_pullback(ȳ)
        Ȳ = ChainRulesCore.unthunk(ȳ)
        Ȳm = Ȳ isa Base.ReshapedArray || Ȳ isa SubArray ? collect(Ȳ) : Ȳ
        wv = _halfdim_weights(f, nh, d, T)
        f̄ = _rfftn(Ȳm) .* (wv ./ N)
        return (ChainRulesCore.NoTangent(), f̄, ChainRulesCore.NoTangent())
    end
    return y, irfftn_pullback
end

function ChainRulesCore.rrule(::typeof(_rfftn), x::AbstractArray{T,3}) where {T<:Real}
    y = _rfftn(x)
    d = size(x, 3); nh = d ÷ 2 + 1
    function rfftn_pullback(ȳ)
        Ȳ = ChainRulesCore.unthunk(ȳ)
        Ȳm = Ȳ isa Base.ReshapedArray || Ȳ isa SubArray ? collect(Ȳ) : Ȳ
        wv = _halfdim_weights(x, nh, d, T)
        x̄ = _brfftn(Ȳm ./ wv, d)
        return (ChainRulesCore.NoTangent(), x̄)
    end
    return y, rfftn_pullback
end
