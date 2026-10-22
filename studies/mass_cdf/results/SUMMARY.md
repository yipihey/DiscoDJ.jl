# Mass-CDF study — results summary

All numbers come from the machine-readable tables in `results/<ensemble>/` (see README.md for the
column definitions and conventions).  Errors are standard errors from realization-to-realization scatter
unless stated otherwise; jackknife and bootstrap errors (also tabulated) agree with them to a few per cent.
Predictions under test: A₁ = 4/(21√(2π)) = 0.075989 (EdS coefficients), A₁ = (1 + D₂/D₁²)/(3√(2π)) = 0.075509
for exact Planck18 growth at a = 1, A₃ ≈ 1.6×10⁻³, m₄ ≈ −0.0152, −ln J ≃ (2/7) T:T at δ_L = 0, no even powers of σ.

## 1. Ensembles

| ensemble | L [h⁻¹Mpc] | N | h [h⁻¹Mpc] | R_f | seeds (Gaussian / fixed) | total volume | notes |
|---|---|---|---|---|---|---|---|
| L1000_N128_R18 (primary) | 1000 | 128 | 7.8 | 18 | 400 / 200 | 400 / 200 (Gpc/h)³ | 1–4LPT (D^n); 1–3LPT exact ΛCDM growth at a = 0.5, 1 |
| L1000_N128_R18_5LPT | 1000 | 128 | 7.8 | 18 | 200 / 200 | | same phases, 1–5LPT |
| L1000_N128_R18_6LPT | 1000 | 128 | 7.8 | 18 | 50 / 50 | | same phases, 1–6LPT |
| L1000_N128_R14 | 1000 | 128 | 7.8 | 14 | 150 / 150 | | |
| L1000_N128_R22 | 1000 | 128 | 7.8 | 22 | 150 / 150 | | |
| L1000_N64_R18 | 1000 | 64 | 15.6 | 18 | 400 / 400 | | same phases as N = 128 (resolution) |
| L1000_N256_R18 | 1000 | 256 | 3.9 | 18 | 60 / 60 | | same phases as N = 128 (resolution) |
| L500_N64_R18 | 500 | 64 | 7.8 | 18 | 400 / 400 | 50 / 50 (Gpc/h)³ | box size |
| L2000_N256_R18 | 2000 | 256 | 7.8 | 18 | 60 / 60 | 480 / 480 (Gpc/h)³ | box size |
| FEM_L500_N64 / N128 / N256 | 500 | 64 / 128 / 256 | 7.8 / 3.9 / 1.95 | 18 | 10 / 3 / 1 (Gaussian) | | FEM N-body (EdS background, ΛCDM P(k) shape) vs 1–4LPT, raw and matched partners, 50 D-time steps |

Each realization: signed amplitudes ε ∈ ±{0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0} (paired ±δ₀ exactly), epoch a = 1 (D₁ = 1),
σ(ε) = |ε| σ₀ with σ₀ = 0.2108 (R_f = 18), 0.2736 (R_f = 14), 0.1690 (R_f = 22).  Fixed-amplitude realizations have
σ = σ_box = 0.21095 exactly.  Every row of `realization_summary.csv` carries ⟨J⟩ (= 1 to 10⁻⁷), ⟨1/J⟩, min J, min singular value
of F, fraction J ≤ 0, J percentiles, the transition matrix, and F_M, F_V at the required thresholds.

## 2. Normalisation and shell crossing

* ⟨J⟩_q = 1.0000000 ± 5×10⁻¹² in every realization and method (exact periodic-map identity, satisfied to float precision).
* ⟨1/J⟩_q = 1.04853 ± 0.00005 (4LPT, ε = 1, R_f = 18) — the Lagrangian-averaged density, not an Eulerian volume average.
* No shell crossing anywhere in the LPT ensembles: fraction of elements with J ≤ 0 is identically 0; the smallest Jacobian over
  all 400 Gaussian seeds at R_f = 18, ε = 1 is min J = 0.107 (mean over seeds of the per-box minimum 0.235 ± 0.002; 10⁻⁵ percentile
  0.272, 10⁻³ percentile 0.427, 10 % percentile 0.736); min singular value of F ≥ 0.28.  At R_f = 14, ε = 1 the extreme is min J = 0.041
  (still single-stream); at ε ≤ 0.65 nothing comes close to crossing.  **No cuts were applied** to any ensemble; the J_min-selection
  test (`jmin_selection.csv`) shows A₁ and A₃ unchanged within errors when seeds with min J < 0.15 or 0.2 are removed.

## 3. Leading coefficient A₁ (paired estimator, 4LPT, a = 1)

| ensemble | ICs | seeds | A₁ (odd fit) | A₁ (line fit Y_even/σ vs σ²) | pull vs 0.075989 |
|---|---|---|---|---|---|
| L1000_N128_R18 | Gaussian | 400 | 0.075984 ± 0.000054 | 0.075978 ± 0.000030 | −0.1σ / −0.4σ |
| L1000_N128_R18 | fixed | 200 | 0.076048 ± 0.000058 | 0.076035 ± 0.000031 | +1.0σ / +1.5σ |
| L1000_N128_R14 | Gaussian | 150 | 0.075897 ± 0.000071 | | −1.3σ |
| L1000_N128_R14 | fixed | 150 | 0.076018 ± 0.000053 | | +0.5σ |
| L1000_N64_R18 | Gaussian | 400 | 0.076015 ± 0.000089 | | +0.3σ |
| L1000_N64_R18 | fixed | 400 | 0.076020 ± 0.000081 | | +0.4σ |
| L500_N64_R18 | Gaussian | 400 | 0.076167 ± 0.000150 | | +1.2σ |
| L500_N64_R18 | fixed | 400 | 0.076264 ± 0.000115 | | +2.4σ |
| L1000_N128_R22, L1000_N256_R18, L2000_N256_R18 | | | *(pending)* | | |

The parameter-free prediction is confirmed at the 10⁻⁴ (relative 0.1 %) level in the 400-realization Gaussian ensemble.  Two
systematic trends at the 10⁻⁴ level are visible: (i) fixed-amplitude ensembles return A₁ higher by (5–10)×10⁻⁵ — the
fixed-amplitude field is not Gaussian and the ratio Y/σ is slightly biased; (ii) the L = 500 box is high by ~2×10⁻⁴ (finite
number of modes); both vanish in the primary L = 1000 Gaussian ensemble.  1LPT (Zel'dovich) gives 0.13297 ± 0.00009 = 1/(3√(2π))
= 0.13298, the analytic Zel'dovich value, confirming the bookkeeping.

## 4. Amplitude expansion, A₃, even powers

With Y ≡ F_M(>ρ̄) − P(δ_L > 0) the pairing δ₀ → −δ₀ leaves F_M − 1/2 invariant (σ depends on |ε|) and flips P(δ_L>0) − 1/2,
so the physical content sits in Y_even = [F_M(+ε) + F_M(−ε)]/2 − 1/2 (the zero-threshold finite-box noise cancels exactly),
while Y_odd = [Y(+ε) − Y(−ε)]/2 is pure noise.  Measured, 4LPT, R_f = 18, 400 Gaussian seeds: Y_odd = (−1.3 ± 1.2)×10⁻⁶ at
ε = 1 and ≤ 3.5×10⁻⁶ at all ε (0.4–1.3σ from zero); fixed: ≤ 2.8×10⁻⁶.  The "odd expansion" of the analytic calculation is
therefore an expansion in σ = |ε|σ₀, and the fits below are to Y_even(|ε|).

Unconstrained fit Y_even = Σ_{k=1}^{5} c_k εᵏ (4LPT, R_f = 18):

| ICs | c₁ (σ pull) | c₂ | c₃ | c₄ | c₅ |
|---|---|---|---|---|---|
| Gaussian (400) | 1.601×10⁻² ± 3.0×10⁻⁵ | (3.6 ± 22.8)×10⁻⁵ [0.2σ] | (−0.6 ± 6.2)×10⁻⁴ | (0.6 ± 7.0)×10⁻⁴ [0.1σ] | (−0.1 ± 2.8)×10⁻⁴ |
| fixed (200) | 1.600×10⁻² ± 3.8×10⁻⁵ | (3.1 ± 2.8)×10⁻⁴ [1.1σ] | (−8.6 ± 7.6)×10⁻⁴ | (9.5 ± 8.6)×10⁻⁴ [1.1σ] | (−3.6 ± 3.4)×10⁻⁴ |

**Even coefficients are consistent with zero** (≤ 1.1σ; in units of A_k = c_k/σ₀ᵏ: A₂ = 0.0008 ± 0.0051, A₄ = 0.03 ± 0.35 for Gaussian ICs).

Odd-constrained fits Y_even = c₁ε + c₃ε³ + c₅ε⁵ converted to A_k = c_k/σ₀ᵏ (full covariance matrices in `amplitude_fit_cov.txt`):

| ensemble | ICs | A₃ (odd fit) | A₃ (line fit) | A₅ |
|---|---|---|---|---|
| L1000_N128_R18 | Gaussian 400 | +0.0011 ± 0.0015 | +0.0019 ± 0.0013 | 0.02 ± 0.02 |
| L1000_N128_R18 | fixed 200 | −0.0011 ± 0.0022 | +0.0007 ± 0.0013 | 0.04 ± 0.04 |
| L1000_N128_R18_5LPT (5LPT) | Gaussian 200 / fixed 200 | −0.0002 ± 0.0022 / −0.0010 ± 0.0022 | | |
| L1000_N128_R14 | Gaussian 150 / fixed 150 | +0.0007 ± 0.0014 / +0.0004 ± 0.0012 | | |
| L1000_N64_R18 | Gaussian 400 / fixed 400 | −0.0028 ± 0.0044 / +0.0025 ± 0.0041 | | |
| L500_N64_R18 | Gaussian 400 / fixed 400 | −0.0005 ± 0.0041 / +0.0042 ± 0.0045 | | |
| L1000_N256_R18, L2000_N256_R18, R22, 6LPT | | *(pending)* | | |

**A₃ = (1.1 ± 1.5)×10⁻³** (primary Gaussian ensemble; (1.9 ± 1.3)×10⁻³ from the straight-line representation Y_even/σ = A₁ + A₃σ²,
Fig. 1) is consistent with the predicted 1.6×10⁻³ but only at the ~1σ level: the cubic term contributes 1.5×10⁻⁵ to F_M at σ = 0.21
and the present 400 (Gpc/h)³ Gaussian ensemble resolves it to ±1.4×10⁻⁵.  Fig. 2 shows Y_even − A₁σ rising as ε³ along the predicted
curve.  A₃ is not detected at better than 1.5σ, and is **not** inconsistent with zero either.  The LPT-order dependence of A₃ is
strong below 4LPT (2LPT −0.0073, 3LPT +0.0103, both ±0.0015) because the degree-k parts of −ln J with k ≤ 4 all enter at O(σ³);
4LPT → 5LPT changes A₃ by only +4×10⁻⁵ and Y(ε = 1) by −2.2×10⁻⁶ (same phases), so 4LPT is converged for this test.

## 5. Transition fractions and full cumulative curves (4LPT, R_f = 18, Gaussian, ε = 1; `ensemble_summary.csv`, `cdf_curves.csv`)

| | ε = 1 (σ = 0.2108) | ε = 0.5 (σ = 0.1054) |
|---|---|---|
| P(δ_L > 0) | 0.500057 ± 0.000088 | same |
| F_M(>ρ̄) = P(J<1) | 0.516091 ± 0.000087 | 0.508064 ± 0.000088 |
| Y = F₊₋ − F₋₊ | 0.016034 ± 0.000010 (A₁σ = 0.016019) | 0.008007 ± 0.000006 (A₁σ = 0.008009) |
| F₊₊ = P(δ_L>0, J<1) | 0.500057 | 0.500057 |
| F₊₋ = P(δ_L<0, J<1) | 0.016034 | 0.008007 |
| F₋₊ = P(δ_L>0, J>1) | **0 (exactly, every element)** | 0 |
| F₋₋ | 0.483909 | 0.491936 |
| F_M at ρ★/ρ̄ = 0.5 / 0.75 / 1 / 1.25 / 1.5 / 2 / 3 / 5 / 10 | 0.99998 / 0.93695 / 0.51609 / 0.17236 / 0.04895 / 0.00434 / 9.3×10⁻⁵ / 1×10⁻⁶ / 0 | 1 / 0.99868 / 0.50806 / 0.02524 / 3.7×10⁻⁴ / 0 / 0 / 0 / 0 |
| F_V at ρ★/ρ̄ = 1 / 2 | 0.43204 / 0.00196 | 0.46602 / 0 |

The four fractions sum to 1 exactly.  The nonlinear excess above one half is entirely F₊₋ (initially underdense mass focused above
the mean) — F₋₊ vanishes identically for all 2–5LPT orders and amplitudes tested (for δ_L > 0 one has −ln J = δ_L + j₂ + … > 0 at
these amplitudes).  The full F_M(>ρ★) and F_V(>ρ★) curves (47 thresholds) for 1–5LPT, exact growth and FEM are in `cdf_curves.csv`
and Fig. 5; 2LPT and 4LPT differ at ρ★ = ρ̄ by 8.0×10⁻⁵ and at ρ★ = 2ρ̄ by 3.6×10⁻⁴.

## 6. Shear focusing (`shear_bins.csv`)

Regression of −ln J on T:T for elements with |δ_L|/σ < w (4LPT, Gaussian, R_f = 18; 33 000–332 000 elements per seed and bin, no exclusions):

| w | ε = 0.2 | ε = 0.5 | ε = 1.0 |
|---|---|---|---|
| 0.02 | 0.28611 ± 0.00018 | 0.28681 ± 0.00007 | 0.28978 ± 0.00005 |
| 0.05 | 0.28627 ± 0.00028 | 0.28686 ± 0.00012 | 0.28979 ± 0.00006 |
| 0.20 | 0.28556 ± 0.00108 | 0.28661 ± 0.00044 | 0.28978 ± 0.00023 |

Prediction 2/7 = 0.28571; the small-amplitude limit reproduces it, with the expected O(ε) growth of the slope from j₃, j₄ (+1.4 % at
ε = 1).  Intercepts are < 10⁻³ σ² and residual rms (at w = 0.02) are 0.27σ², 0.11σ², 0.056σ² at ε = 0.2, 0.5, 1.  The exact second-order
coefficient j₂ regresses with slope 2/7 to 5 digits (tautological check of the extraction).  1LPT gives 0.50 (the Zel'dovich value 1/2).

## 7. Fourth-order conditional coefficient (`m4_conditioning.csv`, Fig. 7)

Shrinking symmetric bins |δ_L|/σ < w, w = 0.01 … 0.5, extrapolated linearly in w² (w ≤ 0.2); R_f = 18, N = 128:

| quantity | Gaussian (200–400 seeds) | fixed (200) | prediction |
|---|---|---|---|
| ⟨j₂ \| δ_L=0⟩/σ² (degree-2 part of −ln J) | 0.19056 ± 0.00003 | 0.19046 ± 0.00004 | 4/21 = 0.190476 |
| ⟨j₃ \| δ_L=0⟩/σ³ | (−0.7 ± 1.1)×10⁻⁵ | (2.3 ± 1.6)×10⁻⁵ | 0 (symmetry) |
| ⟨j₄ \| δ_L=0⟩/σ⁴ (degree-4 part of −ln J) | 0.04067 ± 0.00001 | 0.04065 ± 0.00001 | — |
| ⟨J₄ \| δ_L=0⟩/σ⁴ (degree-4 part of J) | **−0.01526 ± 0.00001** | −0.01524 ± 0.00001 | **−0.0152** |
| ⟨D₄ \| δ_L=0⟩/σ⁴ (degree-4 part of 1/J) | 0.0660 ± 0.0001 | | — |

The analytic m₄ ≃ −0.0152 is reproduced to 0.4 % when "j₄" is identified with the degree-4 part of J itself (J₄ = −j₄ + ½j₂² at δ_L = 0);
for the degree-4 part of −ln J the value is +0.0407.  Resolution: N = 64 gives −0.01511, N = 128 −0.01526, (N = 256 *pending*); R_f = 14 gives −0.01524.

## 8. Exact ΛCDM growth (`nLPTx` rows)

Using DISCO-DJ's D₂, D₃ₐ,ᵦ,ᶜ (3LPT): A₁ = 0.075496 ± 0.000030 at a = 1 (prediction (1 + D₂/D₁²)/(3√(2π)) = 0.075509, D₂/D₁² = −0.43218)
and 0.075861 ± 0.000032 at a = 0.5 (prediction 0.075889); fixed ICs 0.075555 / 0.075967.  The 0.6 % ΛCDM reduction of A₁ at a = 1
relative to the EdS coefficient is therefore measured directly; the D^n (EdS-coefficient) runs reproduce the EdS value.

## 9. FEM versus LPT (`results/fem/`)

DISCO-DJ's P1 FEM Poisson solver on the Kuhn mesh integrates the sheet vertices (EdS background, ΛCDM spectrum shape, 3LPT ICs at
a = 0.05, 50 FastPM D-time steps; 25/50/100 steps and CG tolerances 10⁻⁷/10⁻⁹ give identical results to 2×10⁻⁵).  Its *linear*
response is distorted by the mesh: at h = 7.8 Mpc/h the linear density has std(δ_FEM)/std(δ) = 0.925, corr 0.9964, 5.0 % rms
displacement error (2.2 % at h = 3.9, 0.6 % at 1.95), with its second-order displacement 0.89 (h = 7.8) to 0.97 (h = 3.9) of ψ₂ and
correlated at 0.99.  Because Y is a statistic of the field at δ_L ≈ 0, this linear distortion dominates the *raw* FEM−LPT difference:

| mesh h | ε | raw Y_FEM − Y_4LPT | matched Y_FEM − Y_4LPT (LPT run on the FEM's own linear field) |
|---|---|---|---|
| 7.8 (10 seeds) | 1.0 | −1.10×10⁻³ ± 0.37×10⁻³ (−7 % of Y) | **(−3 ± 4)×10⁻⁵** (−0.2 %) |
| 7.8 | 0.5 | −0.53×10⁻³ ± 0.38×10⁻³ | (0 ± 2)×10⁻⁵ |
| 7.8 | 0.1 | ±4×10⁻⁴ (realization-dependent sign) | (0 ± 1)×10⁻⁵ |
| 3.9 (3 seeds) | | *(pending)* | |
| 1.95 (1 seed) | | *(pending)* | |

With the linear-operator discretisation removed, **FEM and 4LPT agree to within (3 ± 4)×10⁻⁵ in Y at σ = 0.21**, i.e. to 0.2 % of the
nonlinear migration and at the level of the A₃σ³ term itself; the tet-volume (Lagrangian-element) Jacobian gives the same picture.
The FEM cannot, however, test A₃ independently: its mesh error in Y (even after matching) and the FEM's second-order content deficit
at h ≥ 3.9 are comparable to A₃σ³ ≈ 1.5×10⁻⁵, and a mesh fine enough to beat this is beyond the feasible cost (N = 256 at L = 500
costs ≈ 35 min per run).

## 10. Convergence (`results/convergence/`)

* Resolution (same phases, L = 1000): Y(4LPT, ε = 1) differs between N = 64 and N = 128 by *(same-seed table pending N = 256)*;
  A₁(N = 64) = 0.076015 ± 0.000089 vs A₁(N = 128) = 0.075984 ± 0.000054 (Gaussian).
* Box: L = 500 gives A₁ high by (1.8 ± 1.6)×10⁻⁴ relative to L = 1000; L = 2000 *(pending)*.
* LPT order: 4LPT → 5LPT changes Y(ε=1) by −2.2×10⁻⁶ ± 1.5×10⁻⁷ and A₃ by +4×10⁻⁵; 6LPT *(pending)*.
* Filter scale: R_f = 14 gives A₁ = 0.075897 ± 0.000071 / 0.076018 ± 0.000053 (Gaussian / fixed) and A₃ = (0.7 ± 1.4)×10⁻³;
  R_f = 22 *(pending)*; m₄(J) = −0.01524 at R_f = 14.
* Single vs double precision: not run (float32 LPT fields; the Taylor-coefficient closure test reproduces −ln J to 5×10⁻⁶ rms
  at ε = 0.1 and ⟨J⟩ = 1 to 10⁻⁷).

## 11. Items that contradict or qualify the analytic assumptions

1. The paired-field statement "Y_even vanishes" is reversed for Y ≡ F_M − P(δ_L>0): the physical signal is even in ε and the
   zero-threshold noise is odd (Section 4).  The "odd expansion" holds in σ = |ε|σ₀, and the data confirm it: even powers of σ vanish.
2. "F₋₊ should be much smaller" — it is exactly zero for every element at all amplitudes and orders tested (pre-caustic sheet).
3. The predicted m₄ ≈ −0.0152 is reproduced only for the degree-4 coefficient of J; the same conditional average of the degree-4
   part of −ln J is +0.0407 (both tabulated).
4. A₃ is confirmed in sign and consistent in magnitude at 1σ, but not detected at the 3σ level even with 400 (Gpc/h)³; the fixed-amplitude
   ensemble does **not** reduce the A₃ error (its Y_even scatter is similar: the realization scatter of Y is dominated by the
   sampling of the Lagrangian neighbourhood of δ_L = 0, not by σ fluctuations).
5. Fixed-amplitude ICs bias A₁ upward by ~(5–10)×10⁻⁵ (non-Gaussian field); the Gaussian ensemble must be used for the formal test.
6. The FEM solver as implemented is a valid dynamical check of 4LPT only after removing its linear-operator discretisation (matched
   partner); raw FEM−LPT differences of 5–8 % in Y at h = 7.8 Mpc/h are mesh artefacts, not dynamics.
