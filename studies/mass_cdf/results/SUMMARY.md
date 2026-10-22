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
| L1000_N128_R22 | Gaussian | 150 | 0.075978 ± 0.000113 | | −0.1σ |
| L1000_N128_R22 | fixed | 150 | 0.076051 ± 0.000090 | | +0.7σ |
| L1000_N128_R18_6LPT (6LPT) | Gaussian / fixed | 50 / 50 | 0.075909 ± 0.000150 / 0.075935 ± 0.000119 | | −0.5σ / −0.5σ |
| L1000_N256_R18 | Gaussian / fixed | 60 / 60 | 0.075889 ± 0.000122 / 0.075917 ± 0.000085 | | −0.8σ / −0.8σ |
| L2000_N256_R18 | Gaussian / fixed | 60 / 60 | *(filled at the end)* | | |

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
| L1000_N128_R22 | Gaussian 150 / fixed 150 | +0.0059 ± 0.0038 / −0.0017 ± 0.0043 | | |
| L1000_N128_R18_6LPT (4/5/6LPT, same seeds) | Gaussian 50 | −0.00414 / −0.00414 / −0.00413 (± 0.0043) | | |
| L1000_N256_R18 | Gaussian 60 / fixed 60 | +0.0023 ± 0.0014 / +0.0055 ± 0.0013 | +0.0018 ± 0.0031 / +0.0029 ± 0.0022 | |
| L2000_N256_R18 | Gaussian 60 / fixed 60 | *(filled at the end)* | | |

**A₃ = (1.1 ± 1.5)×10⁻³** (primary Gaussian ensemble; (1.9 ± 1.3)×10⁻³ from the straight-line representation Y_even/σ = A₁ + A₃σ²,
Fig. 1) is consistent with the predicted 1.6×10⁻³ but only at the ~1σ level: the cubic term contributes 1.5×10⁻⁵ to F_M at σ = 0.21
and the present 400 (Gpc/h)³ Gaussian ensemble resolves it to ±1.4×10⁻⁵.  Fig. 2 shows Y_even − A₁σ rising as ε³ along the predicted
curve.  A₃ is not detected at better than 1.5σ, and is **not** inconsistent with zero either.  The LPT-order dependence of A₃ is
strong below 4LPT (2LPT −0.0073, 3LPT +0.0103, both ±0.0015) because the degree-k parts of −ln J with k ≤ 4 all enter at O(σ³);
4LPT → 5LPT changes A₃ by only +4×10⁻⁵ and Y(ε = 1) by −2.2×10⁻⁶ (same phases), and 5LPT → 6LPT by 1×10⁻⁵, so 4LPT is converged for this test (the LPT-truncation systematic on A₃ is < 10⁻⁴, 30× below the statistical error).

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
a = 0.05, 50 FastPM D-time steps; 100 steps with CG tolerance 10⁻⁹ instead of 50 steps with 10⁻⁷ changes Y_FEM by 5×10⁻⁵ at ε = 1, i.e. the time-integration systematic is 0.3 % of Y, `FEM_L500_N64_steps100`).  Its *linear*
response is distorted by the mesh: at h = 7.8 Mpc/h the linear density has std(δ_FEM)/std(δ) = 0.925, corr 0.9964, 5.0 % rms
displacement error (2.2 % at h = 3.9, 0.6 % at 1.95), with its second-order displacement 0.89 (h = 7.8) to 0.97 (h = 3.9) of ψ₂ and
correlated at 0.99.  Because Y is a statistic of the field at δ_L ≈ 0, this linear distortion dominates the *raw* FEM−LPT difference:

| mesh h | ε | raw Y_FEM − Y_4LPT | matched Y_FEM − Y_4LPT (LPT run on the FEM's own linear field) |
|---|---|---|---|
| 7.8 (10 seeds) | 1.0 | −1.10×10⁻³ ± 0.37×10⁻³ (−7 % of Y) | **(−3 ± 4)×10⁻⁵** (−0.2 %) |
| 7.8 | 0.5 | −0.53×10⁻³ ± 0.38×10⁻³ | (0 ± 2)×10⁻⁵ |
| 7.8 | 0.1 | ±4×10⁻⁴ (realization-dependent sign) | (0 ± 1)×10⁻⁵ |
| 3.9 (3 seeds) | 1.0 | (−0.9 ± 1.9)×10⁻⁴ (−0.5 %) | **(−4 ± 2)×10⁻⁵** (−0.2 %) |
| 3.9 | 0.5 | (+0.8 ± 1.6)×10⁻⁴ | (−2 ± 1)×10⁻⁵ |
| 3.9 | 0.1 | ±1.4×10⁻⁴ | (−1 ± 2)×10⁻⁵ |
| 1.95 | | not completed: the N = 256 FEM run is single-threaded in JAX on this machine (> 2 h per run) and was stopped | |

The raw difference scales as h² (−1.1×10⁻³ → −0.9×10⁻⁴ from h = 7.8 to 3.9, with the linear density distortion 7.5 % → 2.0 % and the second-order displacement deficit 11 % → 3 %).  With the linear-operator discretisation removed, **FEM and 4LPT agree to within (3–4 ± 2–4)×10⁻⁵ in Y at σ = 0.21 on both meshes** (h²-extrapolated matched difference (−4 ± 3)×10⁻⁵ at ε = 1, (−3 ± 2)×10⁻⁵ at ε = 0.5, (0 ± 2)×10⁻⁵ at ε = 0.1; odd and even parts both ≤ 3×10⁻⁵), i.e. to 0.2 % of the
nonlinear migration and at the level of the A₃σ³ term itself; the tet-volume (Lagrangian-element) Jacobian gives the same picture.
The FEM cannot, however, test A₃ independently: its mesh error in Y (even after matching) and the FEM's second-order content deficit
at h ≥ 3.9 are comparable to A₃σ³ ≈ 1.5×10⁻⁵, and a mesh fine enough to beat this is beyond the feasible cost (N = 256 at L = 500
costs ≈ 35 min per run).

## 10. Convergence (`results/convergence/`)

* Resolution (same phases, L = 1000): Y(4LPT, ε = 1) differs between N = 64 and N = 128 by (−1.3 ± 7.9)×10⁻⁶ (Gaussian, 800 pairs) and
  between N = 128 and N = 256 by (+5 ± 6)×10⁻⁶ (120 pairs), i.e. the grid is converged at the 10⁻⁵ level in Y (≲ 10⁻³ in A₃);
  A₁ = 0.076015 ± 0.000089 / 0.075984 ± 0.000054 / 0.075889 ± 0.000122 at N = 64 / 128 / 256 (Gaussian); m₄(J) = −0.01511 / −0.01526 / −0.01526.
* Box: L = 500 gives A₁ high by (1.8 ± 1.6)×10⁻⁴ relative to L = 1000; L = 2000 *(pending)*.
* LPT order: 4LPT → 5LPT changes Y(ε=1) by −2.2×10⁻⁶ ± 1.5×10⁻⁷ and A₃ by +4×10⁻⁵; 5LPT → 6LPT changes A₃ by 1×10⁻⁵ (50 seeds, same phases). A₁ is identical to 6 digits across 4–6LPT.
* Filter scale: R_f = 14 gives A₁ = 0.075897 ± 0.000071 / 0.076018 ± 0.000053 (Gaussian / fixed) and A₃ = (0.7 ± 1.4)×10⁻³;
  R_f = 22 gives A₁ = 0.075978 ± 0.000113 / 0.076051 ± 0.000090 and A₃ = (5.9 ± 3.8)×10⁻³ / (−1.7 ± 4.3)×10⁻³; m₄(J) = −0.01524 at R_f = 14 and −0.01524 at R_f = 22 (no measurable R_f dependence between 14 and 22).
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

## 12. Single-cube second-order CDF (`results/local_diagnostics/single_cube_cdf.csv`, `_fits.csv`, `fig_single_cube_cdf_residual`)

F_M^(2)(σ) = ∫ p_χ²₅(u) Φ(2gσu/15) du with g = (1 + D₂/D₁²)/2 (EdS 2/7; exact Planck18 growth 0.28391 at a = 1, 0.28534 at a = 0.5),
compared with the measured paired estimator F_M = ½ + Y_even.  F_M^(2) = ½ + A₁σ − 1.158×10⁻³σ³ + … (checked numerically), so the
weak-amplitude intercept of Δ_tide/σ³ should be A₃ + 1.158×10⁻³ ≈ 2.8×10⁻³ if A₃ = 1.6×10⁻³.  Weighted fits Δ_tide/σ³ = a₀ + a₁σ²:

| ensemble | ICs | method | a₀ = A₃,tide | implied A₃ = a₀ − 1.158×10⁻³ |
|---|---|---|---|---|
| L1000_N128_R18 | Gaussian 400 | 4LPT | (1.7 ± 2.0)×10⁻³ | (0.6 ± 2.0)×10⁻³ |
| L1000_N128_R18 | fixed 200 | 4LPT | (5.8 ± 2.8)×10⁻³ | (4.6 ± 2.8)×10⁻³ |
| L1000_N128_R18 | Gaussian 400 | 3LPT exact growth, a = 1 | (11.3 ± 1.9)×10⁻³ | (10.2 ± 1.9)×10⁻³ (3LPT, not converged — cf. §4) |
| L1000_N128_R18_5LPT | Gaussian 200 / fixed 200 | 5LPT | (−0.9 ± 2.8) / (5.8 ± 2.8)×10⁻³ | |
| L1000_N64_R18 | Gaussian 400 / fixed 400 | 4LPT | (0.6 ± 3.2) / (6.5 ± 3.3)×10⁻³ | |
| L1000_N128_R14 | Gaussian 150 / fixed 150 | 4LPT | (−3.3 ± 1.5) / (3.2 ± 1.5)×10⁻³ | |
| L1000_N128_R22 | Gaussian 150 / fixed 150 | 4LPT | (5.3 ± 6.4) / (9.0 ± 6.9)×10⁻³ | |

The residual above the single-cube law is positive and of the expected order (10⁻³ in A₃ units) but, as for A₃ itself, only at the
1–2σ level; this diagnostic is a re-expression of §4 (it is dominated by the same ε³ term) and inherits its noise.  Caution: any A₁
offset (fixed-amplitude bias, finite box at L = 500) enters Δ_tide/σ³ as δA₁/σ² and dominates the small-σ end — the L = 500 rows
(not listed; a₀ ≈ 0.02–0.03) are an artefact of that, and the fixed-amplitude rows are biased upward for the same reason.

## 13. Fourth-order residual at initially mean density (`mean_density_residual*.csv`, `fig_mean_density_residual_and_shape`)

R ≡ −ln J − gD²S and the exact second-order remainder R₂ ≡ −ln J − [δ_L + (13/42)δ_L² + gD²S] for |δ_L|/σ < w, binned in S = T:T and
fitted on elements as R/σ⁴ = c₀ + c₁(S/σ²) + c₂(S/σ²)² (4 Gaussian seeds, R_f = 18, N = 128; ± seed scatter):

| quantity (w = 0.05) | c₀ | c₁ | c₂ | residual rms/σ⁴ |
|---|---|---|---|---|
| exact j₄ (degree-4 part of −ln J, the ε → 0 law) | 0.0042 ± 0.0001 | 0.0217 ± 0.0003 | 0.0356 ± 0.0002 | 0.019 |
| R₂, 4LPT, ε = 1 | 0.0101 ± 0.0013 | 0.0051 ± 0.0042 | 0.0427 ± 0.0027 | 0.22 |
| R₂, 4LPT, ε = 0.5 | 0.0147 ± 0.0026 | −0.0085 ± 0.0084 | 0.0467 ± 0.0054 | 0.43 |
| R₂, 4LPT, ε = 0.2 | 0.030 ± 0.007 | −0.052 ± 0.021 | 0.062 ± 0.014 | 1.07 |
| R₂, 2LPT, ε = 1 | 0.0184 ± 0.0017 | −0.0242 ± 0.0049 | 0.0447 ± 0.0031 | 0.39 |
| R₂, 3LPT exact growth, ε = 1 | 0.0142 ± 0.0012 | 0.0437 ± 0.0040 | 0.0331 ± 0.0027 | 0.22 |
| R₂, FEM (own linear tensor), N = 128, ε = 1 (1 seed) | −0.27 | 0.36 | 0.08 | 0.58 |
| R₂, matched 4LPT, N = 128, ε = 1 (1 seed) | 0.030 | −0.065 | 0.067 | 0.22 |

The clean fourth-order law is the j₄ row: ⟨j₄ | S, δ_L = 0⟩/σ⁴ = 0.0042 + 0.0217 (S/σ²) + 0.0356 (S/σ²)², whose S-average
(⟨S/σ²⟩ = 2/3, ⟨(S/σ²)²⟩ = 28/45) reproduces m₄(−ln J) = 0.0408 of §7.  At finite ε the remainder R₂ is dominated by the third-order
term (ε³, zero mean at δ_L = 0 but large scatter, hence the rms/σ⁴ ∝ 1/ε) so the finite-ε fits are noisy and do not converge
cleanly to the j₄ coefficients within their errors; R (with the δ_L terms left in, as defined in the request) is in addition dominated by
the δ_L spread inside the bin and is useless for this purpose at any w (`target = R` rows).  The FEM at h = 3.9 differs from
matched 4LPT at the level c₀ ≈ −0.3 (i.e. ⟨R₂⟩ ≈ 0.023σ⁴ vs 0.028σ⁴): a 3×10⁻⁴ offset in −ln J at mean density, again the
linear-operator distortion.

## 14. Shear shape beyond T:T (`shear_shape_test*.csv`)

Within |δ_L|/σ < 0.05 and S/σ² bins, R₂ is regressed on χ₃ = tr T³/(T:T)^{3/2} (quadratic in χ₃, linear in S inside the bin).
A clear, highly significant dependence exists; it scales exactly as ε³ (third order), i.e. it is the tr T³ term that second order
cannot contain: d⟨R₂⟩/dχ₃ / σ³ = −0.0247, −0.051, −0.094, −0.163, −0.286 for S/σ² ≈ 0.3, 0.5, 0.75, 1.1, 1.65 (identical at ε = 0.2,
0.5, 1 to < 2 %), i.e. ⟨R₂ | T⟩ ≃ −0.145 tr(T³) (the coefficient −0.145 ± 0.005 is the same in every S bin; the local Zel'dovich
piece alone would give −1/3, so the non-local 2LPT/3LPT contributions supply +0.19).  Fraction of the R₂ residual variance explained
by χ₃ in the cells (all S bins, w = 0.05): 4LPT 0.38–0.39, 2LPT 0.14 (2LPT lacks the third-order tr T³ pieces), matched 4LPT on the
FEM grid 0.25–0.33, FEM itself 0.004–0.04 (its residual is dominated by the mesh linear response).  **The first local tensor
information needed beyond (δ_L, T:T) is tr T³ at third order; it is sizeable per element but, being odd in the field, averages out
of F_M at O(σ²) and only enters the CDF at O(σ³) through its correlation with the fourth-order terms.**

## 15. Multiscale (centred 3h parent) locality diagnostic (`results/multiscale/REPORT.md` and tables; parallel work package)

Decision: **do not build a multiscale parent/child theory.**  The Gaussian new-information fraction f_s^new = 1 − r_s² is 3.4×10⁻³
(h = 7.8, R_f = 18), 8×10⁻³ (R_f = 14), 1.7×10⁻³ (R_f = 22), 2.2×10⁻⁴ (h = 3.9), identical in the three cubic sectors, predicted by
the box sums and measured to < 1 %, scaling as (h/R_f)^{3.5–3.8}.  The literal R_gain of the invariant regressions is 0.65–0.69, but
the one-scale residual is already 3×10⁻⁵–2×10⁻⁴ of Var[ln J], and the gain is a deterministic quadrature effect (cube-averaged tensor
versus the edge-sampled tensor the Kuhn cube actually sees): with the consistent cube tensor A_K the one-scale residual drops 6–11×
and R_gain falls to 0.11–0.13 (ε = 1) / 0.01 (ε = 0.5).  The integrated fractions are unchanged by the parent (|ΔF_M| ≤ 4×10⁻⁵, same
size as the definition effect), and the residual does not respond to the parent-conditioned innovations (|corr| ≤ 0.03).
