"""
KernelAbstractions.jl implementations of the LPT compute kernels.

Every kernel here has a matching Threads.@threads twin in kernels_threads.jl.
The KA versions work on any backend (CPU(), CUDABackend(), ROCBackend(), …).

Kernels:
- `inv_laplace_ka!`     — divide Fourier field by k² (Poisson solve)
- `grad_multiply_ka!`   — multiply by i·k[d] (Fourier gradient)
- `fmu2_elementwise_ka!`— point-wise product of two Fourier fields (fmu2 building block)
- `field_add_ka!`       — accumulate a contribution into an output field
- `real_to_complex_ka!` — cast real field → complex in-place
"""

export inv_laplace_ka!, grad_multiply_ka!, fmu2_elementwise_ka!,
       field_add_ka!, apply_growth_ka!

using KernelAbstractions

# ── inv_laplace_ka! ───────────────────────────────────────────────────────────

@kernel function _inv_laplace_kernel!(out, @Const(f), @Const(k2))
    i = @index(Global, Linear)
    @inbounds out[i] = k2[i] == 0 ? zero(eltype(out)) : f[i] / k2[i]
end

"""
    inv_laplace_ka!(out, f, k2; backend=CPU())

Compute out[i] = f[i] / k2[i]  (zeros at k=0).
Solves the Poisson equation:  φ(k) = δ(k)/k².
"""
function inv_laplace_ka!(out::AbstractArray{T}, f::AbstractArray{T},
                         k2::AbstractArray; backend=CPU()) where T
    kernel = _inv_laplace_kernel!(backend)
    kernel(out, f, k2, ndrange=length(out))
    KernelAbstractions.synchronize(backend)
    return out
end

# ── grad_multiply_ka! ─────────────────────────────────────────────────────────

@kernel function _grad_kernel!(out, @Const(fphi), @Const(kcomp))
    i = @index(Global, Linear)
    @inbounds out[i] = im * kcomp[i] * fphi[i]
end

"""
    grad_multiply_ka!(out, fphi, kcomp; backend=CPU())

Compute out[i] = i * kcomp[i] * fphi[i].
Used to take the Fourier-space gradient: ψ_d(k) = i·k_d·φ(k).
"""
function grad_multiply_ka!(out::AbstractArray{Complex{T}}, fphi::AbstractArray{Complex{T}},
                           kcomp::AbstractArray; backend=CPU()) where T
    kernel = _grad_kernel!(backend)
    kernel(out, fphi, kcomp, ndrange=length(out))
    KernelAbstractions.synchronize(backend)
    return out
end

# ── fmu2_elementwise_ka! ──────────────────────────────────────────────────────
# The 2LPT fmu2 kernel (Algorithm 1, arXiv:2010.12584) requires products of
# first-order potential derivatives. The most expensive step is point-wise
# complex multiplication in Fourier space.

@kernel function _fmu2_kernel!(out, @Const(f1), @Const(f2))
    i = @index(Global, Linear)
    @inbounds out[i] += f1[i] * f2[i]
end

"""
    fmu2_elementwise_ka!(out, f1, f2; backend=CPU())

Accumulate out[i] += f1[i] * f2[i] (complex Fourier-space product).
Used to build the 2LPT source term from products of ψ¹ derivatives.
"""
function fmu2_elementwise_ka!(out::AbstractArray{Complex{T}},
                              f1::AbstractArray{Complex{T}},
                              f2::AbstractArray{Complex{T}};
                              backend=CPU()) where T
    kernel = _fmu2_kernel!(backend)
    kernel(out, f1, f2, ndrange=length(out))
    KernelAbstractions.synchronize(backend)
    return out
end

# ── field_add_ka! ─────────────────────────────────────────────────────────────

@kernel function _field_add_kernel!(out, @Const(f), scale)
    i = @index(Global, Linear)
    @inbounds out[i] += scale * f[i]
end

"""
    field_add_ka!(out, f, scale; backend=CPU())

Compute out[i] += scale * f[i].
"""
function field_add_ka!(out::AbstractArray{T}, f::AbstractArray{T},
                       scale::Number; backend=CPU()) where T
    kernel = _field_add_kernel!(backend)
    kernel(out, f, T(scale), ndrange=length(out))
    KernelAbstractions.synchronize(backend)
    return out
end

# ── apply_growth_ka! ──────────────────────────────────────────────────────────

@kernel function _apply_growth_kernel!(psi, @Const(psi1), @Const(psi2),
                                       D1::T, D2::T) where T
    i = @index(Global, Linear)
    @inbounds psi[i] = D1 * psi1[i] + D2 * psi2[i]
end

"""
    apply_growth_ka!(psi, psi1, psi2, D1, D2; backend=CPU())

Combine first- and second-order displacements at given growth factors:
ψ(a) = D₁(a)·ψ₁ + D₂(a)·ψ₂
"""
function apply_growth_ka!(psi::AbstractArray{T}, psi1::AbstractArray{T},
                          psi2::AbstractArray{T}, D1::Number, D2::Number;
                          backend=CPU()) where T
    kernel = _apply_growth_kernel!(backend)
    kernel(psi, psi1, psi2, T(D1), T(D2), ndrange=length(psi))
    KernelAbstractions.synchronize(backend)
    return psi
end
