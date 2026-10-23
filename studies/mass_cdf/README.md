# Mass-CDF study: Lagrangian mass fraction above the mean density on a pre-shell-crossing sheet

High-statistics numerical tests of the parameter-free analytic prediction

    F_M(>ρ̄) = P(J < 1) = 1/2 + A₁ σ + A₃ σ³ + …,      A₁ = 4/(21√(2π)) = 0.075989…  (EdS growing-mode coefficients)

for a smoothed, globally single-stream dark-matter sheet, together with the transition
fractions, the full mass/volume CDFs, the shear-focusing regression −ln J vs T:T near
δ_L = 0, the odd-power amplitude expansion, and the direct fourth-order conditional
coefficient m₄ = ⟨j₄ | δ_L = 0⟩/σ⁴.  Everything physical comes from DISCO-DJ (the
`disco-dj-fem` checkout wrapped by `DiscoDJLib`): filtered ΛCDM P(k), Gaussian ICs, 1–4LPT,
exact ΛCDM growth functions (2LPT/3LPT), and the FEM (Kuhn-tessellation P1 Poisson) N-body.
No physics code was modified.  All results tables are machine readable (`results/`).

## Layout

```
studies/mass_cdf/
  Project.toml            Julia environment (DiscoDJLib via path, PythonCall, JLD2, CairoMakie, …)
  src/MassCDF.jl          module: bridge.jl (DISCO-DJ calls), whitenoise.jl (matched-phase ICs),
                          stats.jl (per-element statistics kernel), ensemble.jl (LPT driver),
                          fem.jl (FEM vs LPT driver), analysis.jl (aggregation, fits, errors)
  bin/run_ensemble.jl     worker: <config.toml> <gaussian|fixed> <seed_lo> <seed_hi>
  bin/run_fem.jl          worker: <config.toml> <seed_lo> <seed_hi>
  bin/analyze.jl          aggregate one ensemble into results/<name>/*.csv
  bin/analyze_fem.jl      aggregate the FEM ensembles into results/fem/*.csv
  bin/make_figures.jl     figures into results/figures/
  bin/diag_single_cube.jl single-cube second-order CDF residual (SUMMARY §12)
  bin/diag_local.jl, bin/diag_local_summary.jl   mean-density fourth-order residual and shear-shape test (§13–14)
  bin/run_smallscale.jl, bin/analyze_smallscale.jl   small R_f / post-shell-crossing test with PM N-body reference (§16)
  multiscale/             centred 3h parent/child locality diagnostic (results/multiscale/REPORT.md, §15)
  bin/stage_launch.sh     the worker launch commands used for the campaign (stages 1–3)
  bin/finalize.sh         waits for the last ensembles, then runs every analysis script
  config/*.toml           ensemble definitions (cosmology, box, resolution, filter, amplitudes, …)
  data/<name>/            worker outputs (JLD2; element-level dumps *_elements_seed*.jld2 are large and not committed)
  results/<name>/         tables (large per-realization tables gzipped); results/figures/; results/fem/;
                          results/convergence/; results/SUMMARY.md (the written-up results)
```

Run (from the repository root; the Python interpreter is DISCO-DJ's venv, resolved by `DiscoDJLib`):

```sh
julia -t 4 --project=studies/mass_cdf studies/mass_cdf/bin/run_ensemble.jl studies/mass_cdf/config/L1000_N128_R18.toml gaussian 1 100
julia -t 4 --project=studies/mass_cdf studies/mass_cdf/bin/run_fem.jl      studies/mass_cdf/config/FEM_L500_N128.toml 1 3
julia      --project=studies/mass_cdf studies/mass_cdf/bin/analyze.jl L1000_N128_R18
julia      --project=studies/mass_cdf studies/mass_cdf/bin/make_figures.jl L1000_N128_R18 L1000_N64_R18 L1000_N256_R18 L500_N64_R18 L2000_N256_R18
```

## Conventions (important for comparing with the analytic calculation)

* **Filter.** W(k) = exp(−k²R_f²/2) acts on δ, i.e. P_f(k) = P(k)·exp(−k²R_f²).  This is realised
  with DISCO-DJ's own cutoff, `with_linear_ps(k0_cutoff = 1/(√2·R_f), cutoff_type = "gaussian")`
  (DISCO-DJ multiplies P by exp(−½(k/k0)²)).  Note that the repository's earlier PM suite quoted
  "R = 1/k_cut", which corresponds to R_f = R/√2 in the present convention.  For Planck18EEBAOSN
  (Eisenstein–Hu transfer, σ₈ = 0.8105) at z = 0: σ(R_f = 14, 18, 22 h⁻¹Mpc) = 0.274, 0.211, 0.169.
* **σ.** Per realization, σ is the rms of the realised filtered linear density (`sigma`);
  `sigma_box` is the discrete-mode expectation Σ_k P_f/L³ and `sigma_cont` the continuum integral.
  Fixed-amplitude realizations have σ = σ_box exactly.
* **Epoch and growth.** The D^n-LPT family ("Dn") is ψ = Σ (εD₁)ⁿ ψₙ with EdS coefficients and
  D₁ = D₊(a) of the exact cosmology, evaluated at a = 1 (D₁ = 1); because the fields depend on the
  linear field only through εD₁, an epoch sequence is equivalent to the amplitude sequence.  The
  exact-growth family ("exact", methods `nLPTx`, n ≤ 3) uses DISCO-DJ's D₂(a), D₃ₐ,ᵦ,ᶜ(a) at
  a ∈ {0.5, 1}; the corresponding leading coefficient is A₁ = (1 + D₂/D₁²)/(3√(2π)) =
  0.075509 (a = 1), 0.075889 (a = 0.5) versus 0.075989 in EdS.
* **Amplitude sequence.** δ₀ → εδ₀ with ε ∈ ±{0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0}; the paired field
  −δ₀ is ε → −ε (ψₙ → (−1)ⁿψₙ).  Y ≡ F_M(>ρ̄) − P(δ_L > 0) per realization.
* **Matched phases.** One master white-noise field (unit variance, N_fine³ = 256³) per seed; coarser
  grids keep its modes below their Nyquist frequency, so N = 64/128/256 at fixed L share phases and
  resolution convergence is a same-realization difference.  Fixed-amplitude ICs normalise every mode
  to |δ_k|² = P_f(k) (Angulo & Pontzen 2016); Gaussian and fixed realizations of the same seed share phases.
* **Jacobian.** F = I + ∂ψ/∂q from DISCO-DJ's `evaluate_jacobian_from_psi` (exact ik kernel, Nyquist
  plane zeroed); J = det F at the N³ Lagrangian grid points, equal-mass elements.  No density
  deposition anywhere.  ⟨J⟩_q = 1 is verified per realization (`meanJ`).
* **Taylor coefficients.** jₖ ≡ degree-k part of −ln J in ε from the trace–log expansion of
  det(I + Σ εⁿGₙ) (exact per element, no fitting):  j₁ = δ_L,  j₂ = −tr G₂ + ½tr G₁²,
  j₃ = −tr G₃ + tr(G₁G₂) − ⅓tr G₁³,  j₄ = −tr G₄ + ½tr G₂² + tr(G₁G₃) − tr(G₁²G₂) + ¼tr G₁⁴.
  ⟨j₂ | δ_L = 0⟩/σ² = (2/7)(2/3) = 4/21 in EdS.  The conditional-moment tables carry three variants:
  `lnJ` (the jₖ above), `J` (degree-k parts of J = exp(−Σεᵏjₖ): J₂ = −j₂ + ½δ_L², J₄ = −j₄ + δ_L j₃ + ½j₂² − ½δ_L²j₂ + δ_L⁴/24, …)
  and `invJ` (degree-k parts of 1/J).  The analytic m₄ ≃ −0.0152 refers to the `J` variant (⟨J₄ | δ_L = 0⟩/σ⁴).
* **FEM.** `nbody_fem.fem_nbody_cosmo` (FastPM-style D-time stepping, P1 FEM Poisson solve on the
  deformed Kuhn mesh, tet-wise mass) in an EdS background with the Planck18 P(k) shape, from 3LPT
  ICs at a_ini = 0.05 to a = 1.  Each seed is also run at ±ε_lin = ±0.1; the ε-odd part gives the
  FEM's own linear displacement, whose divergence defines the "matched" linear field on which the
  LPT partner is rerun (`with_external_ics(delta=…)`).  See `results/fem/` and SUMMARY.md.

## Tables (per ensemble, `results/<name>/`)

* `realization_summary.csv` — one row per realization × method × epoch × signed ε: σ's, P(δ_L>0),
  F_M(>ρ̄), Y, ⟨J⟩, ⟨1/J⟩, ⟨ln J⟩, min J, max J, fraction J ≤ 0, min singular value of F, J percentiles
  (1e-5 … 0.1), F₊₊, F₊₋, F₋₊, F₋₋, F_M and F_V at ρ★/ρ̄ ∈ {0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5, 10}.
* `ensemble_summary.csv` — means and standard errors (realization scatter) of the above per family × ε.
* `cdf_curves.csv` — F_M(>ρ★), F_V(>ρ★) on the full threshold grid (47 values, 10^−1 … 10^1.3) with errors.
* `shear_bins.csv` (+ `_per_seed`) — regression of −ln J on T:T in |δ_L|/σ < {0.02, 0.05, 0.1, 0.2}:
  slope, intercept, residual rms, counts; also the exact-j₂ regression (ε → 0 limit).
* `amplitude_fit.csv` (+ `_per_seed.csv`, `_cov.txt`, `_cov.jld2`) — fits Y(ε): unconstrained
  c₁…c₅ ("U"), odd-constrained c₁,c₃,c₅ ("O"), odd two-term on |ε| ≤ 0.65 ("O3"); per-seed
  coefficients, ensemble means with realization-scatter / jackknife / bootstrap errors, GLS fit to the
  ensemble-mean curve, full covariance matrices; A_k = c_k/σ₀ᵏ.
* `odd_even.csv`, `yodd_line_fit.csv` — paired decomposition Y_odd, Y_even per ε and the line fit
  Y_odd/σ = A₁ + A₃σ² (+A₅σ⁴).
* `m4_conditioning.csv` (+ `_per_seed`) — ⟨jₖ⟩/σᵏ in shrinking symmetric bins and the w² → 0 extrapolation.
* `meta.toml` — cosmology, box, resolution, filter, seeds, code commit hashes.
