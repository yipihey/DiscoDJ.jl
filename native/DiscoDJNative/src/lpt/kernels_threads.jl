"""
Threads.@threads implementations of the LPT compute kernels.

Every function here is a direct twin of a KA kernel in kernels_ka.jl.
The signatures are identical so callers can switch by passing `backend=:threads`.

From GraphGP.jl experience: KA on CPU can have higher overhead per-element
than raw @threads because of the kernel-launch indirection.
These implementations test whether that's true for our specific kernels.
"""

export inv_laplace_threads!, grad_multiply_threads!, fmu2_elementwise_threads!,
       field_add_threads!, apply_growth_threads!

# ── inv_laplace_threads! ──────────────────────────────────────────────────────

"""
    inv_laplace_threads!(out, f, k2)

Threaded Poisson solve: out[i] = f[i] / k2[i]  (zero at k=0).
"""
function inv_laplace_threads!(out::AbstractArray{T}, f::AbstractArray{T},
                              k2::AbstractArray) where T
    Threads.@threads for i in eachindex(out)
        @inbounds out[i] = k2[i] == 0 ? zero(T) : f[i] / k2[i]
    end
    return out
end

# ── grad_multiply_threads! ────────────────────────────────────────────────────

"""
    grad_multiply_threads!(out, fphi, kcomp)

Threaded Fourier gradient: out[i] = i * kcomp[i] * fphi[i].
"""
function grad_multiply_threads!(out::AbstractArray{Complex{T}},
                                fphi::AbstractArray{Complex{T}},
                                kcomp::AbstractArray) where T
    Threads.@threads for i in eachindex(out)
        @inbounds out[i] = im * kcomp[i] * fphi[i]
    end
    return out
end

# ── fmu2_elementwise_threads! ─────────────────────────────────────────────────

"""
    fmu2_elementwise_threads!(out, f1, f2)

Threaded accumulate: out[i] += f1[i] * f2[i].
"""
function fmu2_elementwise_threads!(out::AbstractArray{Complex{T}},
                                   f1::AbstractArray{Complex{T}},
                                   f2::AbstractArray{Complex{T}}) where T
    Threads.@threads for i in eachindex(out)
        @inbounds out[i] += f1[i] * f2[i]
    end
    return out
end

# ── field_add_threads! ────────────────────────────────────────────────────────

"""
    field_add_threads!(out, f, scale)

Threaded accumulate: out[i] += scale * f[i].
"""
function field_add_threads!(out::AbstractArray{T}, f::AbstractArray{T},
                            scale::Number) where T
    s = T(scale)
    Threads.@threads for i in eachindex(out)
        @inbounds out[i] += s * f[i]
    end
    return out
end

# ── apply_growth_threads! ─────────────────────────────────────────────────────

"""
    apply_growth_threads!(psi, psi1, psi2, D1, D2)

Threaded growth application: psi[i] = D1*psi1[i] + D2*psi2[i].
"""
function apply_growth_threads!(psi::AbstractArray{T}, psi1::AbstractArray{T},
                               psi2::AbstractArray{T}, D1::Number, D2::Number) where T
    d1 = T(D1); d2 = T(D2)
    Threads.@threads for i in eachindex(psi)
        @inbounds psi[i] = d1 * psi1[i] + d2 * psi2[i]
    end
    return psi
end
