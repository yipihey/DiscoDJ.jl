"""
Differentiable nLPT displacement — `lpt_psi_ad`.

`lpt_psi_ad(fphi, grid, cosmo, a; n_order, exact_growth)` returns the
growth-weighted displacement ψ(a) as a `(res,res,res,3)` real array, differentiable
w.r.t. the initial potential `fphi` (and hence the white-noise field ω via the
faithful `white_noise_to_fphi`).  This is the property that makes DISCO-DJ "Done
with Jax" — the prerequisite for field-level IC inference.

It is now a thin wrapper over the **faithful, de-aliased** general-order engine
(`nlpt_core.jl`): `compute_core`/`compute_core_exact` are written functionally
(plain `rfft`/`irfft` + `cat`, no in-place mutation) so the AbstractFFTs ChainRules
differentiate straight through, and they reproduce the JAX reference to machine
precision.  (The earlier bespoke `lpt_psi_ad` was a longitudinal-only, non-de-aliased
EdS approximation in the opposite +1/k² gauge; it is superseded.)

Pair with `ic_operator`/`white_noise_to_fphi` (the JAX −1/k² gauge), NOT with the
legacy `generate_grf`/`compute_lpt` (+1/k² gauge).
"""

export lpt_psi_ad

using ChainRulesCore: @ignore_derivatives

"""
    lpt_psi_ad(fphi, grid, cosmo, a; n_order=2, exact_growth=false) -> (res,res,res,3)

Differentiable displacement ψ(a) from the initial Fourier potential `fphi`
(JAX −1/k² gauge, e.g. from `white_noise_to_fphi`).  Backed by the faithful
`compute_core`/`compute_core_exact`; `grid` supplies the resolution and box size.
"""
function lpt_psi_ad(fphi::AbstractArray{Complex{T},3}, grid::FourierGrid{T},
                    cosmo::Cosmology, a::Real; n_order::Int=2,
                    exact_growth::Bool=false) where {T}
    # Kernels are constants in ω (and contain `setindex!` Nyquist-zeroing) — build
    # them off the AD tape.  For repeated calls, prefer precomputing `nlpt_kernels`
    # once and calling `lpt_displacement` directly.
    K = @ignore_derivatives nlpt_kernels(grid.res, grid.boxsize; T=T)
    return lpt_displacement(fphi, K, cosmo, a; n_order, exact_growth)
end
