"""
Differentiable nLPT displacement field.

`lpt_psi_ad(fphi, grid, cosmo, a; n_order)` returns the growth-weighted displacement
ψ(a) = Σ_n D_n(a)·ψ_n as a `(res,res,res,3)` real array — the same quantity as
`evaluate_lpt_psi_at_a(compute_lpt(fphi, grid; n_order), cosmo, a)`, but written
*functionally* (plain `rfft`/`irfft` + broadcasts, no in-place `mul!`, buffer reuse,
or `unsafe_free!`).  That makes it traceable by Zygote/ChainRules: the FFTs carry
AbstractFFTs' ChainRules and everything else is a differentiable broadcast, so the
gradient w.r.t. the initial potential `fphi` (and hence the white-noise field ω, via
`white_noise_to_fphi`) comes out with no custom adjoint.

The optimised, memory-lean `compute_lpt` stays the forward-only path; this is the
inference/gradient path (mirrors how JAX DISCO-DJ is written).
"""

export lpt_psi_ad

using ChainRulesCore: @ignore_derivatives   # growth factors / k² are constants in ω

# rfft/irfft in the pipeline's [3,1,2] convention (half-axis on dim 3).
_rfft312(x)       = rfft(x, [3, 1, 2])
_irfft312(fx, res) = irfft(fx, res, [3, 1, 2])

function lpt_psi_ad(fphi::AbstractArray{Complex{T},3}, grid::FourierGrid{T},
                    cosmo::Cosmology, a::Real; n_order::Int=2) where {T}
    n_order in (1, 2, 3) || error("n_order must be 1, 2, or 3")
    kx, ky, kz = grid.k_vecs
    res = grid.res
    # 1/k² and the growth factors do not depend on ω — keep them out of the AD tape
    # (Zygote's pullback compilation segfaults tracing growth_D1's ODE integration).
    invk2 = @ignore_derivatives(@. ifelse(grid.k2 == 0, zero(T), inv(grid.k2)))

    grad(fc, kc) = _irfft312((im .* kc) .* fc, res)          # ψ_d = irfft(i·k_d·field)
    sderiv(fc, ki, kj) = _irfft312((-(ki .* kj)) .* fc, res) # φ,ij in real space

    # 1LPT
    psi1 = cat(grad(fphi, kx), grad(fphi, ky), grad(fphi, kz); dims = 4)
    D1   = @ignore_derivatives T(growth_D1(cosmo, a))
    psi  = D1 .* psi1

    if n_order >= 2
        d11 = sderiv(fphi, kx, kx); d22 = sderiv(fphi, ky, ky); d33 = sderiv(fphi, kz, kz)
        d12 = sderiv(fphi, kx, ky); d13 = sderiv(fphi, kx, kz); d23 = sderiv(fphi, ky, kz)
        S2  = @. d11*d22 - d12^2 + d11*d33 - d13^2 + d22*d33 - d23^2
        fphi2 = (T(-3/7) .* _rfft312(S2)) .* invk2
        psi2  = cat(grad(fphi2, kx), grad(fphi2, ky), grad(fphi2, kz); dims = 4)
        D2    = @ignore_derivatives(T(growth_D2(cosmo, a)) * D1^2)
        psi   = psi .+ D2 .* psi2
    end

    if n_order >= 3
        d2_11 = sderiv(fphi2, kx, kx); d2_22 = sderiv(fphi2, ky, ky); d2_33 = sderiv(fphi2, kz, kz)
        d2_12 = sderiv(fphi2, kx, ky); d2_13 = sderiv(fphi2, kx, kz); d2_23 = sderiv(fphi2, ky, kz)
        S3a = @. d11*(d22*d33 - d23^2) - d12*(d12*d33 - d23*d13) + d13*(d12*d23 - d22*d13)
        S3b = @. d2_11*(d22 + d33) + d2_22*(d11 + d33) + d2_33*(d11 + d22) -
                 2*(d2_12*d12 + d2_13*d13 + d2_23*d23)
        S3   = @. T(10/21)*S3a + T(1/3)*S3b
        fphi3 = (T(-1) .* _rfft312(S3)) .* invk2
        psi3  = cat(grad(fphi3, kx), grad(fphi3, ky), grad(fphi3, kz); dims = 4)
        D3    = @ignore_derivatives D1^3
        psi   = psi .+ (D3 * T(1/3)) .* psi3
    end

    return psi
end
