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
       evaluate_core, lpt_displacement

using ChainRulesCore: @ignore_derivatives   # growth factors are constants in ω

# ── rfft/irfft in the pipeline's [3,1,2] convention ───────────────────────────
_rfftn(x) = rfft(x, [3, 1, 2])
_irfftn(f, n::Int) = irfft(f, n, [3, 1, 2])

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
struct NLPTKernels{T<:AbstractFloat}
    res::Int
    ext::Int          # extended (de-aliasing) resolution = 3·res÷2
    boxsize::T
    d_dx::Array{Complex{T},3}; d_dy::Array{Complex{T},3}; d_dz::Array{Complex{T},3}
    inv_lap::Array{T,3}
    dx_ext::Array{Complex{T},3}; dy_ext::Array{Complex{T},3}; dz_ext::Array{Complex{T},3}
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
    Z = Complex{T}
    # dim 1 (full): [lo | zeros | hi]
    a1 = cat(ff[1:h, :, :], zeros(Z, ext - 2h + 1, orig, h + 1),
             ff[h+2:orig, :, :]; dims=1)                              # (ext, orig, h+1)
    # dim 2 (full)
    a2 = cat(a1[:, 1:h, :], zeros(Z, ext, ext - 2h + 1, h + 1),
             a1[:, h+2:orig, :]; dims=2)                              # (ext, ext, h+1)
    # dim 3 (half): only the lower block, zero-pad up to ext÷2+1
    a3 = cat(a2[:, :, 1:h], zeros(Z, ext, ext, ext ÷ 2 + 1 - h); dims=3)
    return a3 .* T((ext / orig)^3)
end

"""Crop an extended rfft field back to `(orig,orig,orig÷2+1)` (inverse block map of
`_pad3`, Nyquist planes → 0), with the (orig/ext)³ amplitude rescale."""
function _crop3(ff::AbstractArray{Complex{T},3}, orig::Int, ext::Int) where {T}
    h = orig ÷ 2
    Z = Complex{T}
    nz = size(ff, 3)
    # dim 1 (full): [lo | Nyquist=0 | hi]
    a1 = cat(ff[1:h, :, :], zeros(Z, 1, ext, nz), ff[ext-h+2:ext, :, :]; dims=1)   # (orig, ext, nz)
    # dim 2 (full)
    a2 = cat(a1[:, 1:h, :], zeros(Z, orig, 1, nz), a1[:, ext-h+2:ext, :]; dims=2)  # (orig, orig, nz)
    # dim 3 (half): lower block + zero Nyquist plane
    a3 = cat(a2[:, :, 1:h], zeros(Z, orig, orig, orig ÷ 2 + 1 - h); dims=3)
    return a3 .* T((orig / ext)^3)
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
    acc = zeros(Complex{T}, res, res, res ÷ 2 + 1)
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
    z() = zeros(Complex{T}, res, res, res ÷ 2 + 1)
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
    acc = zeros(Complex{T}, res, res, res ÷ 2 + 1)
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
        fL = zeros(Complex{T}, res, res, res ÷ 2 + 1)
        fT = zeros(Complex{T}, res, res, res ÷ 2 + 1, 3)

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

    out = Dict{String,Array{T,4}}()
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

    out = Dict{String,Array{T,4}}()
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
function evaluate_core(shapes::Dict{String,Array{T,4}}, cosmo::Cosmology, a::Real;
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
