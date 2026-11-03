# Report B: measuring the two-loop effective stress

Reference: Anastasiou, Favorito, Lewandowski, Senatore & Zheng, arXiv:2509.05187. Design and equation mapping: `twoloop/README.md`. Scripts: `twoloop/tl_measure.jl`, `tl_fit3.py` (with `test_tl_fit3.py`), `tl_closure.py`, `time_local_closure.py`, `tl_even_test.py`, `stage7_analysis.py`, `final_figures.py`. Results: `results/tl_fit3_*`, `tl_closure_*`, `time_local_closure.json`, `stage7_summary.json`. Figures: `figT1_running`, `figT2_stress_time`, `figT3_time_local`.

## Setup

EdS, 4 seeds × ±L, fixed amplitudes. Spectra sharing all modes below k = 0.3: k < 1.6, 1.0, 0.6, and a UV-divergent spectrum (P ∝ k⁻¹ above k = 0.3) with k < 1.6 and 1.0. IR members cut at Λ = 0.1, 0.15, 0.2, 0.3. Stress τ̂ = kinetic + gravitational from 2³-resampled particles, CIC on 384³, sharp coarse-graining. The estimator is τ̂[full] − τ̂[IR cut at Λ], which removes the tree-level coarse-graining of long modes.

**Deposit.** Direct particle sums against the deposit (seed 1, s = +1, D = 0.1–1; tl_full, tl_ir020, tl_ir010): every moment agrees to ≤ 1.1 × 10⁻⁴; full − IR is good to ≤ 2.5 × 10⁻⁴ with tl_ir010, and with tl_ir020 to ≤ 1.5 × 10⁻³ except density at D = 0.1 (3.7 %) and 0.3 (0.9 %). tl_ir015, tl_ir030, composite stress and gravitational stress were not checked directly.

## Results

**UV test of the linear coefficient.** At D = 0.1, c̄/I from band differences, far from Λ, over Λ = 0.1–0.3, against 61/70 = 0.871:

| band | I | stress: PM | stress: 3LPT | stress: 7LPT | density, PM | momentum, PM |
|---|---|---|---|---|---|---|
| k = 0.6–1.6 | 3.58 | 0.863–0.871 | 0.833–0.843 | 0.861–0.869 | 0.878–0.946 | 0.906–0.913 |
| k = 1.0–1.6 | 1.18 | 0.879–0.888 | 0.835–0.846 | 0.874–0.883 | 0.926–0.981 | 0.925–0.933 |
| divergent, k = 1.0–1.6 | 6.21 | 1.040–1.047 | 0.930–0.939 | 1.054–1.062 | 0.959–1.000 | 1.059–1.074 |

For the Planck-shape bands PM's stress is within 1–2 % of 61/70. Density and momentum sit 1–13 % higher, so the channels differ even here. In the divergent band stress is 20 % high, and 7LPT shows the same, so the excess is in single-stream dynamics; at D = 0.1 that spectrum may already be beyond the leading term (not tested). One epoch.

**Running across Λ is not a UV test** (figT1). dc̄/dI over Λ = 0.1–0.3 is 0.844–0.846 in the stress (density 0.89–0.92) in all five spectra, against 0.871. But I(Λ) differences are identical across spectra and the stress differences nearly so, because those modes (k = 0.1–0.3) are shared. So this agreement is built in. It is one measurement, of modes that are not far above Λ, and it is 3 % below 61/70.

**PM against LPT** (figT2). c̄/I at Λ = 0.1, k < 1.6: PM 0.851, 0.918, 1.009, 1.066, 1.088, 1.088 at D = 0.1–0.6; 7LPT 0.850, 0.922, 1.152, 2.16, 5.90, 12.1. They agree to D ≈ 0.2, then PM levels off. In the divergent spectrum PM peaks at D ≈ 0.2 (0.957) and falls to 0.80 by D = 0.6. In the far-UV band k = 0.6–1.6, PM's coefficient falls to about 0.06 by D = 0.5, though the linear fit explains only 11 % of the variance there.

**Closure with a power series.** With c(D) = c₀ + c₂D²: c₀ agrees across stress, density and momentum to 0–6 %, c₂ to 16–27 %, for k < 1.6. c₀ is 0.80–0.88. It fails for the divergent spectrum, where stress gives c₂ ≈ 0 against 5.9–11.8 in density.

**Is the stress local in time?** (figT3). This is the better closure. The measured c(D) passes through y″ + (3/2D)y′ − (3/2D²)y = D c(D); a stress local in time and linear then fixes the density and momentum responses, with no free parameter:

| Λ = 0.15 | D | stress | density: predicted, measured | momentum: predicted, measured |
|---|---|---|---|---|
| k < 1.6 | 0.3 | 1.091 | 0.935, 0.947 | 0.992, 1.016 |
| | 0.5 | 1.202 | 1.043, 1.087 | 1.119, 1.169 |
| k < 0.6 | 0.3 | 1.057 | 0.913, 0.897 | 0.963, 0.932 |
| | 0.5 | 1.413 | 1.056, 1.033 | 1.195, 1.165 |
| divergent, k < 1.0 | 0.3 | 1.120 | 0.975, 1.007 | 1.036, 1.104 |
| | 0.5 | 1.005 | 1.039, 1.126 | 1.046, 1.146 |
| divergent, k < 1.6 | 0.3 | 0.919 | 0.948, 1.096 | 0.950, 1.136 |
| | 0.5 | 0.818 | 0.916, 1.080 | 0.878, 1.007 |

(Measured density here is the direct k²-fit coefficient 9C₂/D² at k ≤ 0.08; momentum is tl_fit3's fixed-epoch sum.) Prediction relative to measured, to D = 0.5: Planck-shape spectra ≤ 4.4 % at Λ = 0.15 and ≤ 8.5 % at Λ = 0.1 (mostly a constant offset already at D = 0.1, and larger against fixed-epoch density, up to 7.7 % at Λ = 0.15); divergent k < 1.0, 3–9 %; divergent k < 1.6, 8–13 % at D = 0.2 and 13–17 % from D = 0.3. Either a linear, time-local stress does not describe the regime or the fitted linear coefficient misses what acts on the long modes; **this test cannot separate the two.** It uses linear interpolation between epochs and constant c below D = 0.1.

**Direct cross-check with report A's line.** The k²-fit density coefficient in full − IR at Λ = 0.2 gives C₂ = 0.538 ± 0.003 at D = 0.5 and 1.922 ± 0.020 at D = 1 (4 seeds), against 0.532 ± 0.007 and 1.995 ± 0.015 from the random-long run (k_uv = 1.2) at k ≤ 0.08.

**Stochastic part.** The stress-trace residual is white at k ≲ 0.13: slope −0.2 to +0.07 (odd), 0 to +0.47 (even) for PM. PM and 7LPT agree to D = 0.2, after which 7LPT grows by orders of magnitude and steepens.

## What could not be measured

* **Second- and third-order coefficients.** Stress is usable only at D ≤ 0.3 for Λ = 0.2 and D ≤ 0.6 for Λ = 0.1, because 1/n_Λ of a sharp filter is singular beyond. In windows D ≤ 0.3:
  * c_δ,1 + c_r,1 is determined.
  * c_rδ,1 is measured but scattered: 1.7–2.2 for k < 1.6 at Λ ≥ 0.2 (analytic 2.04), 1.1–2.5 in far-UV bands, −0.9 in the divergent band.
  * δ,2 and δ²,1 are separately unconstrained. Their sum drifts through the analytic 3.64 with no plateau (far-UV band, 3LPT: 0.46 ± 0.35, 1.51 ± 0.24, 3.98 ± 0.23 at D = 0.2, 0.3, 0.4), and the even stress is noise-dominated at D ≤ 0.2 and contaminated from D ≈ 0.3.
  * δ,3, rδ,2, rδ²,1 are nearly degenerate (condition number ≈ 6 × 10⁶, density-only ≈ 5 × 10¹⁷).

  So those κ are neither confirmed nor refuted, and **the "all eight are needed" test (T7) is undecided**. Windowed jackknife errors understate the uncertainty, because c grows by about 35 % inside the window; they should not be quoted.
* **T3 through c_O,α/c_O,1:** undetermined for the same reason. The time-local test above is what the data can support.
* **T4, fluid-element structure, and T8, Λ-independence of the predicted P(k):** not attempted. T8 needs the full two-loop kernels at field level.
* **T5, vorticity:** rms curl v_Λ against rms k·π_Λ is 0.42, 0.89, 1.48, 2.9 % at D = 0.1–0.4 (Λ = 0.2), with PM and 7LPT together to D = 0.3 for k < 1.6. Above D = 0.4 v_Λ = π_Λ/ρ_Λ is singular and the ratio is not physical. The prediction from τ̃₍₂₎ was not computed.

## Recommendations

A better second-order measurement needs a filter without the n_Λ singularity (a Gaussian or a mass-weighted velocity) and fits with a measured c(D) rather than constant windows. Near the deposit limit, D = 0.1 even blocks are noise. These would be new measurements on the existing snapshots, about 4 hours of CPU at the current pipeline speed.
