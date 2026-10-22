"""
    MassCDF

High-statistics numerical tests of the parameter-free analytic prediction for the
Lagrangian mass fraction above a final Eulerian density threshold,

    F_M(>ρ★) = P(J < ρ̄/ρ★),      F_M(>ρ̄) = P(J < 1) = 1/2 + A₁σ + A₃σ³ + …,

on a smoothed, globally single-stream dark-matter sheet.  Everything physical
(filtered ΛCDM P(k), Gaussian ICs, 1–4LPT, exact ΛCDM growth functions, FEM
N-body) comes from DISCO-DJ through the `DiscoDJLib` bridge; this module only
adds (i) matched-phase white noise (Gaussian / fixed-amplitude, paired ±,
truncatable across resolutions), (ii) the per-element statistics of the
Jacobian field, and (iii) the ensemble bookkeeping, fits and figures.

Conventions (see README.md):
* filter: W(k) = exp(-k²R_f²/2) on δ, i.e. P_f(k) = P(k) exp(-k²R_f²), realised
  through DISCO-DJ's `with_linear_ps(k0_cutoff = 1/(√2 R_f), cutoff_type="gaussian")`;
* σ ≡ rms of the filtered linear density at the epoch (per realization and ensemble);
* D^n-LPT ("Dn" growth): ψ = Σ (εD)ⁿ ψₙ with EdS coefficients, D = D₊(a) of the
  exact cosmology; exact growth ("exact"): ψ = εD₁ψ₁ + ε²D₂ψ₂ᵉˣ + ε³(D₃ₐψ₃ₐ+D₃bψ₃b+D₃cψ₃c);
* jₖ ≡ degree-k homogeneous part of −ln J in the amplitude ε (so −ln J = Σ εᵏ jₖ);
  j₁ = δ_L, j₂ = (13/42)δ_L² + (2/7) T:T for EdS coefficients.
"""
module MassCDF

using PythonCall, LinearAlgebra, Statistics, StaticArrays, FFTW, Random, Printf, Dates, TOML, JLD2
using DataFrames, StatsBase
using DiscoDJLib

export study_config, load_config, run_ensemble, run_fem_ensemble

const A1_EDS = 4 / (21 * sqrt(2π))

include("bridge.jl")
include("whitenoise.jl")
include("stats.jl")
include("ensemble.jl")
include("fem.jl")
include("analysis.jl")

end # module
