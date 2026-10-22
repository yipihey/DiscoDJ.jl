# Multiscale (cube / 3x3x3 parent) locality diagnostic: REPORT

Branch `mass-cdf-study`.  Code in `studies/mass_cdf/multiscale/` (`src/MS.jl`, `src/Regress.jl`, `bin/*.jl`), tables/figures in this directory,
per-cube arrays in `studies/mass_cdf/data/multiscale/` (gitignored).  Nothing under `src/`, `bin/`, `config/`, `Project.toml`, or the existing results was touched.
All numbers below are read from the CSVs in this directory (`scale_correlations.csv`, `locality_gain.csv`, `cdf_multiscale.csv`, `innovation_response.csv`,
`diag_quadrature.csv`, `sanity_lnJ_coefficients.csv`, `oneScale_coefficients.csv`).

## 1. Decision

**Do not build a multiscale (parent/child) theory.  A one-cube local state (the cube-averaged initial deformation tensor A_C) is sufficient for the mass-CDF question.**

* The Gaussian cross-correlation between cube and 3x3x3-parent tensors is almost perfect: 1 - r_s^2 = 3.4e-3 at (h = 7.8, R_f = 18), 8e-3 at R_f = 14, 1.7e-3 at R_f = 22, 2e-4 at h = 3.9; predicted by box sums and measured to <1 % (Sec. 3).
* Taken literally, the requested figure of merit R_gain = 1 - V_CP/V_C is large (0.65-0.69 for 2LPT/4LPT with the cube-averaged tensor), i.e. **above the 10 % threshold**.  However (Sec. 4-5):
  (a) the one-scale residual is already tiny: V_C/Var[ln J] = 3e-5 ... 2.3e-4 (rms residual 1e-3 ... 4e-3 against an rms ln J of 0.17-0.28), so the "gain" removes ~1e-4 of the variance;
  (b) the gain is a **deterministic finite-h quadrature effect**, not environment information: at linear order the Kuhn-cube volume samples the gradient on cube *edges* (vertex differences, corner-sampled transversally) whereas A_C is the cube *average*; the difference is ~ (h^2/12) x curvature of A, which A_P - A_C measures.  A pure Gaussian linear-level calculation predicts R_gain = 0.690 / 0.691 / 0.691 for R_f = 14 / 18 / 22 with optimal weights (0.89, 0.11) on (tr A_C, tr A_P); the measured values are 0.65 / 0.67 / 0.67 (4LPT) and 0.68 / 0.69 / 0.69 (2LPT), and a fit at eps = 0.1 returns weights (0.897, 0.103).  Replacing A_C by the tensor A_K that the Kuhn cube actually sees (edge differences, same Fourier machinery) cuts the one-scale residual variance by a factor 6-11 (4LPT) and leaves **R_gain = 0.11-0.13 (4LPT, eps = 1), 0.01 (eps = 0.5)**; the remaining absolute residual V_CP/Var is 3e-6 ... 3e-5;
  (c) the integrated CDF fractions do not materially change: |F_M(pred) - F_M(actual)| is <= 1.8e-4 (R_f = 14) and ~(1-5)e-5 (R_f = 18, 22) with the one-scale model, against F_M - 1/2 = 0.0157 (R_f = 18) - 0.0207 (R_f = 14); the parent changes this by at most ~4e-5 and has the same sign/size as the statistical scatter or the A_C-vs-A_K definition effect (Sec. 6);
  (d) innovation response of the residual to the parent-conditioned innovations is nil (|corr| <~ 0.02 for the Gaussian cases; Sec. 7).
* For the FEM the parent does not rescue the part that matters: with the FEM's own linear tensor the one-scale model works as for LPT (Sec. 5), the excess FEM residual over matched-4LPT is **not** explained by the parent (R_gain 0.33-0.45 vs 0.6-0.67 for LPT), and with the *input* linear tensor the residual is dominated by the mesh-distorted linear response (rms 5e-3 ... 1.7e-2), which is a linear-operator (transfer-function) effect, not an environment effect.

Caveat on the criterion: if the intended precision on F_M were below ~1e-4 the parent/A_K corrections would matter numerically, but they are removable by a definition (cube tensor) rather than a new state variable.

## 2. Definitions (what was computed)

* Cube (i,j,k) has anchor vertex (i,j,k) and centre (i+1/2,j+1/2,k+1/2)h.  A_C = exact cubic top-hat average of the vertex deformation tensor A = G1 = dpsi_1/dq (band-limited, Nyquist planes zero): Fourier factor prod_i sinc(k_i h/2) times the phase exp(i k.h/2) (FFTW, Float64, 6 independent components).  A_P likewise with prod sinc(3 k_i h/2).  All N^3 cubes, periodic wrap.
* eps scaling: A -> eps A.  Cube Jacobian J = (sum of the six Kuhn-tet volumes)/h^3 from the unwrapped positions q + sum_n eps^n psi_n (4LPT; 2LPT uses n <= 2), via `discodj.analysis.phase_space_sheet.kuhn_cube_volumes_from_positions`.  The same definition is used for LPT and FEM (FEM: positions q + psi_fem; identical to the sum of the stored `Jtet_FEM` over the six tets to 3e-6 (N=64) / 7e-6 (N=128)).
* Sectors: S0 = U + 2V (trace), S_E = U - V (diagonal traceless), S_T = 2V (off-diagonal), with U = int P_f W_a W_b khat_x^4, V = int P_f W_a W_b khat_x^2 khat_y^2.  r_s = S_s^PC/sqrt(S_s^PP S_s^CC), f_s^new = 1 - r_s^2, beta_s = S_s^PC/S_s^PP.  Box sum: rfft grid of the (L,N) box (Nyquist planes and k = 0 excluded, same tabulated P_f as DISCO-DJ); continuum: spherical quadrature in the first octant x 8 (log-k trapezoid 3000 pts x 40x40 Gauss-Legendre angles).
* Regressions (least squares, no tuning, column-scaled normal equations with a 1e-10 ridge):
  * one-scale basis (11 terms; polynomial in the invariants of eps A_C up to degree 4 in A): `1, I1, I2, I3, I1^2, I1*I2, I1^3, I1*I3, I2^2, I1^2*I2, I1^4`
  * two-scale basis (26 terms) = the 11 above (C invariants) + 15 parent/cross terms: `I1P, I2P, I3P, tr(AC AP), tr(AC^2 AP), tr(AC AP^2), I1C*I1P, I1C*I2P, I2C*I1P, I1P^2, I1P^3, I1C*I1P^2, I1C^2*I1P, tr(AC AP)*I1C, tr(AC AP)*I1P`; I1 = tr A, I2 = (I1^2 - tr A^2)/2, I3 = det A.
  * flexible upper bound ("flexible", clearly secondary): all monomials of degree <= 3 in the 6 components of eps A_C (84 terms) versus in the 12 components of eps (A_C, A_P) (455 terms); computed for 4LPT and FEM only, cut J > 0.1.  A raw-component cubic polynomial is not an upper bound in the strict sense: it is not rotation covariant and it extrapolates badly in the ln J tails (e.g. the ln J flexible CDF rows are off by -7.5e-4 in F_M), so use it only as a cross-check of the invariant fit.
  * Targets ln J and J; cuts none (ln J: J > 0; J: all cubes), J > 0.1, J > 0.2 (the cut is applied to train and to the quoted residual statistics; the CDF bookkeeping always uses all test cubes).
* Gaussian cases: L = 1000, N = 128 (h = 7.81), R_f = 14, 18, 22 (eps = 1) and R_f = 18 (eps = 0.5, 1); train seeds 1-4, test seeds 5-8 (8.39M cubes each); R_gain_err = std over the 4 test seeds of the per-seed R_gain / sqrt(4); V_C, V_CP, Var_y are pooled-test variances.
* FEM cases (one seed): spatial split of the same periodic cube set: train = slab i in [0, N/2 - m), test = four sub-slabs of [N/2, N - m) with margin m = ceil(2R_f/h) (5 cubes at N=64, 10 at N=128); the quoted error is the scatter among those 4 sub-slabs and is therefore only indicative.  A_P is always computed on the full periodic box.  A source `fem_lin` = gradient of `psi_lin_fem` (symmetrised; antisymmetric fraction 8e-6), `input` = G1 of the input linear field.  Partners: `4LPT matched`/`2LPT matched` = LPT rerun on delta_fem_lin (A from fem_lin), `raw` = LPT on the input field (A from input).

## 3. Gaussian scale correlations (`scale_correlations.csv`, `fig_fnew_vs_Rf.png`)

f_new = 1 - r_s^2 (trace / E / T sectors), box-sum prediction, versus the value measured on the realizations (seeds 1-8; 1-4 at N=256; 8 seeds at L=500).  Predicted continuum and box values agree to <= 0.3 % in f (see CSV).

| R_f | N | h | h/R_f | f_new box (0/E/T) | f measured (0/E/T) | beta (0/E/T) |
|---|---|---|---|---|---|---|
| 14 | 64 | 15.6 | 1.12 | 9.05e-02 / 9.69e-02 / 8.63e-02 | 9.07e-02 / 9.73e-02 / 8.63e-02 | 1.3036 / 1.3031 / 1.3039 |
| 18 | 64 | 15.6 | 0.868 | 4.41e-02 / 4.64e-02 / 4.27e-02 | 4.42e-02 / 4.65e-02 / 4.27e-02 | 1.2238 / 1.2245 / 1.2234 |
| 22 | 64 | 15.6 | 0.71 | 2.35e-02 / 2.43e-02 / 2.29e-02 | 2.36e-02 / 2.44e-02 / 2.30e-02 | 1.1678 / 1.1682 / 1.1675 |
| 14 | 128 | 7.81 | 0.558 | 8.14e-03 / 8.32e-03 / 8.02e-03 | 8.16e-03 / 8.36e-03 / 8.02e-03 | 1.0961 / 1.0964 / 1.0959 |
| 18 | 128 | 7.81 | 0.434 | 3.42e-03 / 3.46e-03 / 3.38e-03 | 3.42e-03 / 3.48e-03 / 3.38e-03 | 1.0645 / 1.0646 / 1.0644 |
| 22 | 128 | 7.81 | 0.355 | 1.69e-03 / 1.71e-03 / 1.68e-03 | 1.69e-03 / 1.71e-03 / 1.68e-03 | 1.0461 / 1.0461 / 1.0461 |
| 14 | 256 | 3.91 | 0.279 | 5.57e-04 / 5.61e-04 / 5.55e-04 | n/a | 1.0254 / 1.0255 / 1.0254 |
| 18 | 256 | 3.91 | 0.217 | 2.25e-04 / 2.26e-04 / 2.24e-04 | 2.26e-04 / 2.28e-04 / 2.25e-04 | 1.0167 / 1.0167 / 1.0167 |
| 22 | 256 | 3.91 | 0.178 | 1.09e-04 / 1.10e-04 / 1.09e-04 | n/a | 1.0118 / 1.0118 / 1.0118 |
| 18 (L=500, EdS) | 64 | 7.81 | 0.434 | 3.41e-03 / 3.47e-03 / 3.37e-03 | 3.47e-03 / 3.52e-03 / 3.43e-03 | 1.0647 / 1.0637 / 1.0653 |
| 18 (L=500, EdS) | 128 | 3.91 | 0.217 | 2.25e-04 / 2.26e-04 / 2.24e-04 | 2.29e-04 / 2.30e-04 / 2.27e-04 | 1.0167 / 1.0165 / 1.0169 |
| 18 (FEM lin. response, seed 1) | 64 | 7.81 | 0.434 | - | 2.72e-03 / 2.78e-03 / 2.67e-03 | - |
| 18 (FEM lin. response, seed 1) | 128 | 3.91 | 0.217 | - | 2.15e-04 / 2.23e-04 / 2.18e-04 | - |


* The sector spectra are equal to within 1 % in f (cubic symmetry), r is the same for the three sectors: r_s ~ 1 - f/2.
* Box-sum variance check (measured / predicted second moments of the filtered fields; the rows show agreement to 0.6 % for L=1000 (E sector systematically 0.994 for all eight seeds, trace/T 0.997-1.001), and 1-3 % for the 500 Mpc/h box (fewer modes, 8 seeds)):

| L | N | R_f | n seeds | S_CC meas/box (0/E/T) | S_PP meas/box | S_CP meas/box |
|---|---|---|---|---|---|---|
| 1000 | 64 | 14 | 8 | 0.998 / 0.994 / 1.001 | 0.997 / 0.994 / 1.000 | 0.998 / 0.994 / 1.000 |
| 1000 | 64 | 18 | 8 | 0.998 / 0.994 / 1.000 | 0.997 / 0.994 / 0.998 | 0.997 / 0.994 / 0.999 |
| 1000 | 64 | 22 | 8 | 0.997 / 0.994 / 0.999 | 0.996 / 0.995 / 0.997 | 0.997 / 0.994 / 0.998 |
| 1000 | 128 | 14 | 8 | 0.999 / 0.995 / 1.001 | 0.998 / 0.994 / 1.001 | 0.998 / 0.994 / 1.001 |
| 1000 | 128 | 18 | 8 | 0.998 / 0.994 / 1.000 | 0.997 / 0.994 / 1.000 | 0.998 / 0.994 / 1.000 |
| 1000 | 128 | 22 | 8 | 0.997 / 0.994 / 0.999 | 0.997 / 0.994 / 0.998 | 0.997 / 0.994 / 0.999 |
| 1000 | 256 | 18 | 4 | 1.004 / 0.999 / 1.008 | 1.005 / 0.999 / 1.009 | 1.005 / 0.999 / 1.008 |
| 500 | 64 | 18 | 8 | 1.014 / 1.028 / 1.004 | 1.017 / 1.034 / 1.006 | 1.015 / 1.031 / 1.005 |
| 500 | 128 | 18 | 8 | 1.013 / 1.027 / 1.004 | 1.014 / 1.029 / 1.004 | 1.014 / 1.028 / 1.004 |


* Scaling: local log-log slopes of f_new against h/R_f from the box predictions: R_f scan at h = 7.8: p = 3.5; R_f scan at h = 3.9: p = 3.6; R_f scan at h = 15.6: p = 3.0; h scan at R_f = 18: p = 3.8.  So f_new ~ (h/R_f)^p with p approaching the expected 4 only for h/R_f << 0.4 (the sinc expansion is not yet asymptotic at h/R_f ~ 0.4-1).
* FEM: the linear tensor derived from the FEM `psi_lin` is smoother than the input (f_new = 2.7e-3 vs 3.4e-3 at N = 64; 2.1e-4 vs 2.2e-4 at N = 128), consistent with the mesh-distorted linear response (rms 0.194 vs 0.2097 of the input delta at N=64).
* beta_s = S^PC/S^PP is > 1 (1.06 at h = 7.8, R_f = 18) because the parent filter is wider; this is the coefficient used for the sector-wise innovations eta = A_C - beta A_P.

## 4. Locality gain (`locality_gain.csv`, `fig_Rgain_vs_Rf.png`, `fig_residual_rms.png`)

Gaussian ensembles, invariants basis, target ln J, cut J > 0.1 (other targets/cuts are in the CSV and agree: R_gain(J) within +-0.016 of R_gain(ln J); the cuts none / 0.1 / 0.2 change R_gain by <= 0.02 in every Gaussian case):

| case | R_f | eps | Var[ln J] | V_C | V_CP | R_gain (test) | R_gain (train) | V_C/Var | V_CP/Var | R_gain flexible (deg 3) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4LPT (Gaussian, L=1000, N=128) | 14 | 1 | 0.079 | 1.49e-05 | 5.26e-06 | 0.647 +- 0.003 | 0.653 | 1.9e-04 | 6.7e-05 | 0.51 |
| 4LPT (Gaussian, L=1000, N=128) | 18 | 1 | 0.0454 | 3.24e-06 | 1.08e-06 | 0.665 +- 0.003 | 0.670 | 7.1e-05 | 2.4e-05 | 0.59 |
| 4LPT (Gaussian, L=1000, N=128) | 22 | 1 | 0.0286 | 9.61e-07 | 3.19e-07 | 0.668 +- 0.001 | 0.672 | 3.4e-05 | 1.1e-05 | 0.64 |
| 4LPT (Gaussian, L=1000, N=128) | 18 | 0.5 | 0.0108 | 6.47e-07 | 2.3e-07 | 0.645 +- 0.002 | 0.646 | 6.0e-05 | 2.1e-05 | 0.67 |
| 2LPT (Gaussian, L=1000, N=128) | 14 | 1 | 0.0779 | 1.83e-05 | 5.82e-06 | 0.682 +- 0.002 | 0.686 | 2.3e-04 | 7.5e-05 | - |
| 2LPT (Gaussian, L=1000, N=128) | 18 | 1 | 0.0451 | 4.07e-06 | 1.27e-06 | 0.688 +- 0.002 | 0.691 | 9.0e-05 | 2.8e-05 | - |
| 2LPT (Gaussian, L=1000, N=128) | 22 | 1 | 0.0285 | 1.2e-06 | 3.73e-07 | 0.688 +- 0.001 | 0.692 | 4.2e-05 | 1.3e-05 | - |
| 2LPT (Gaussian, L=1000, N=128) | 18 | 0.5 | 0.0108 | 6.82e-07 | 2.32e-07 | 0.659 +- 0.002 | 0.662 | 6.3e-05 | 2.1e-05 | - |


Reading: V_C, V_CP are the test variances of the residual of ln J (rms 1.8e-3 -> 1.0e-3 at R_f = 18, eps = 1); Var[ln J] is 0.045.  R_gain is the same within errors for 2LPT and 4LPT, for ln J and J, and independent of R_f (0.647 ... 0.688); it is slightly smaller at eps = 0.5 (0.645-0.659).  The train values (0.65-0.69) coincide with the test values, i.e. there is no over-fit.  The flexible (cubic raw-component) variant gives a smaller R_gain (0.51-0.67) because the cubic polynomial itself leaves a larger one-scale residual structure in the tails; it is shown only as a cross-check.

### 4.1 Why the gain is not environment information (`diag_quadrature.csv`, `sanity_*`)

At linear order the Kuhn cube sees J_lin(k) = -delta_k sum_a khat_a^2 sinc(k_a h/2) prod_{b != a} cos(k_b h/2) (edge differences), while tr A_C(k) = -delta_k sum_a khat_a^2 prod_b sinc(k_b h/2).  Regressing J_lin on tr A_C alone and on (tr A_C, tr A_P) with the Gaussian box sums (L = 1000, N = 128):

| R_f | R_gain at linear level (Gaussian box sums) | optimal coefficients on (tr A_C, tr A_P) | resid/var, C only | resid/var, C+P |
|---|---|---|---|---|
| 14 | 0.690 | 0.888 / 0.113 | 1.3e-04 | 3.9e-05 |
| 18 | 0.691 | 0.892 / 0.108 | 5.1e-05 | 1.6e-05 |
| 22 | 0.691 | 0.895 / 0.106 | 2.5e-05 | 7.8e-06 |


and, at tiny amplitude, a direct fit of ln J (4LPT, seed 1, R_f = 18) on (I1C, I1P) returns

| eps (seed 1, R_f=18) | fitted (c_I1C, c_I1P) in ln J ~ 1 + c I1C + c I1P |
|---|---|
| 0.1 | 0.8969 / 0.1028 |
| 0.2 | 0.9021 / 0.0976 |
| 0.5 | 0.9187 / 0.0840 |


(the linear-level prediction is (0.892, 0.108)).  So the "parent" merely supplies the curvature of A needed to convert a cube average into the edge-sampled value that the cube volume actually uses.  Consistently, re-doing the regression with the edge-sampled tensor A_K (symmetrised; formula A_K,ab(k) = i k_a psi_b sinc(k_a h/2) prod_{c != a} cos(k_c h/2) e^{i k.h/2}; A_P unchanged) gives:

| method | R_f | eps | cube tensor | V_C (one-scale) | V_CP (two-scale) | R_gain | V_C/Var | V_CP/Var |
|---|---|---|---|---|---|---|---|---|
| 4LPT | 14 | 1 | A_C (cube average) | 1.49e-05 | 5.26e-06 | 0.647 +- 0.003 | 1.9e-04 | 6.7e-05 |
| 4LPT | 14 | 1 | A_K (Kuhn edge tensor) | 2.52e-06 | 2.2e-06 | 0.129 +- 0.004 | 3.2e-05 | 2.8e-05 |
| 4LPT | 18 | 1 | A_C (cube average) | 3.24e-06 | 1.08e-06 | 0.665 +- 0.003 | 7.1e-05 | 2.4e-05 |
| 4LPT | 18 | 1 | A_K (Kuhn edge tensor) | 4.17e-07 | 3.63e-07 | 0.129 +- 0.006 | 9.2e-06 | 8.0e-06 |
| 4LPT | 18 | 0.5 | A_C (cube average) | 6.47e-07 | 2.3e-07 | 0.645 +- 0.002 | 6.0e-05 | 2.1e-05 |
| 4LPT | 18 | 0.5 | A_K (Kuhn edge tensor) | 5.68e-08 | 5.61e-08 | 0.012 +- 0.001 | 5.2e-06 | 5.2e-06 |
| 4LPT | 22 | 1 | A_C (cube average) | 9.61e-07 | 3.19e-07 | 0.668 +- 0.001 | 3.4e-05 | 1.1e-05 |
| 4LPT | 22 | 1 | A_K (Kuhn edge tensor) | 1.09e-07 | 9.66e-08 | 0.115 +- 0.007 | 3.8e-06 | 3.4e-06 |
| 2LPT | 14 | 1 | A_C (cube average) | 1.83e-05 | 5.82e-06 | 0.682 +- 0.002 | 2.3e-04 | 7.5e-05 |
| 2LPT | 14 | 1 | A_K (Kuhn edge tensor) | 4.01e-06 | 2.87e-06 | 0.285 +- 0.005 | 5.1e-05 | 3.7e-05 |
| 2LPT | 18 | 1 | A_C (cube average) | 4.07e-06 | 1.27e-06 | 0.688 +- 0.002 | 9.0e-05 | 2.8e-05 |
| 2LPT | 18 | 1 | A_K (Kuhn edge tensor) | 7.87e-07 | 5.65e-07 | 0.282 +- 0.006 | 1.7e-05 | 1.3e-05 |
| 2LPT | 18 | 0.5 | A_C (cube average) | 6.82e-07 | 2.32e-07 | 0.659 +- 0.002 | 6.3e-05 | 2.1e-05 |
| 2LPT | 18 | 0.5 | A_K (Kuhn edge tensor) | 6.26e-08 | 5.93e-08 | 0.052 +- 0.003 | 5.8e-06 | 5.5e-06 |
| 2LPT | 22 | 1 | A_C (cube average) | 1.2e-06 | 3.73e-07 | 0.688 +- 0.001 | 4.2e-05 | 1.3e-05 |
| 2LPT | 22 | 1 | A_K (Kuhn edge tensor) | 2.1e-07 | 1.54e-07 | 0.265 +- 0.008 | 7.4e-06 | 5.4e-06 |


With A_K the one-scale residual variance drops by 4-12x (e.g. 3.24e-6 -> 4.2e-7 at R_f = 18, eps = 1) and the parent's *relative* gain falls to 11-13 % (4LPT, eps = 1; 28 % for 2LPT, where the residual is the 2LPT-vs-exact-cube difference) and 1-5 % at eps = 0.5; the best two-scale model on A_C (V_CP = 1.1e-6) is still worse than the one-scale model on A_K (4.2e-7).  The remaining 11-13 % (V_CP/Var 8e-6) is a pure O(eps^3-4) sub-cube nonlinearity and is not a parent effect that grows with R_f or h/R_f in any clear way.

### 4.2 FEM (seed 1, L = 500)

| N | eps | target | A source | Var[ln J] | V_C | V_CP | R_gain (test) | R_gain (train) | flexible R_gain |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 1 | FEM | fem_lin | 0.035 | 3.52e-06 | 1.92e-06 | 0.454 +- 0.006 | 0.506 | 0.45 |
| 64 | 1 | FEM | input | 0.035 | 0.000306 | 0.000114 | 0.627 +- 0.009 | 0.699 | 0.98 |
| 64 | 1 | 4LPT matched | fem_lin | 0.035 | 1.91e-06 | 6.34e-07 | 0.668 +- 0.006 | 0.699 | 0.59 |
| 64 | 1 | 4LPT raw | input | 0.0418 | 3.11e-06 | 1.09e-06 | 0.651 +- 0.008 | 0.687 | - |
| 64 | 1 | 2LPT matched | fem_lin | 0.0348 | 2.39e-06 | 7.38e-07 | 0.691 +- 0.007 | 0.710 | - |
| 64 | -1 | FEM | fem_lin | 0.0361 | 3.43e-06 | 2e-06 | 0.417 +- 0.016 | 0.488 | 0.43 |
| 64 | -1 | FEM | input | 0.0361 | 0.000305 | 0.000113 | 0.629 +- 0.007 | 0.707 | 0.97 |
| 64 | -1 | 4LPT matched | fem_lin | 0.0362 | 1.9e-06 | 6.19e-07 | 0.675 +- 0.005 | 0.704 | 0.55 |
| 64 | -1 | 4LPT raw | input | 0.0432 | 3.13e-06 | 1.06e-06 | 0.661 +- 0.004 | 0.691 | - |
| 64 | -1 | 2LPT matched | fem_lin | 0.0359 | 2.37e-06 | 7.23e-07 | 0.694 +- 0.004 | 0.714 | - |
| 64 | 0.5 | FEM | fem_lin | 0.0085 | 5.07e-07 | 2.27e-07 | 0.552 +- 0.003 | 0.599 | 0.56 |
| 64 | 0.5 | FEM | input | 0.0085 | 7.24e-05 | 2.71e-05 | 0.625 +- 0.008 | 0.698 | 0.98 |
| 64 | 0.5 | 4LPT matched | fem_lin | 0.0085 | 3.98e-07 | 1.42e-07 | 0.643 +- 0.004 | 0.679 | 0.63 |
| 64 | 0.5 | 4LPT raw | input | 0.0101 | 6.43e-07 | 2.41e-07 | 0.626 +- 0.006 | 0.665 | - |
| 64 | 0.5 | 2LPT matched | fem_lin | 0.00848 | 4.18e-07 | 1.43e-07 | 0.658 +- 0.004 | 0.692 | - |
| 64 | -0.5 | FEM | fem_lin | 0.00863 | 4.97e-07 | 2.36e-07 | 0.525 +- 0.010 | 0.582 | 0.55 |
| 64 | -0.5 | FEM | input | 0.00863 | 7.23e-05 | 2.7e-05 | 0.626 +- 0.006 | 0.702 | 0.98 |
| 64 | -0.5 | 4LPT matched | fem_lin | 0.00863 | 3.95e-07 | 1.39e-07 | 0.648 +- 0.003 | 0.682 | 0.64 |
| 64 | -0.5 | 4LPT raw | input | 0.0102 | 6.39e-07 | 2.35e-07 | 0.632 +- 0.004 | 0.669 | - |
| 64 | -0.5 | 2LPT matched | fem_lin | 0.00862 | 4.14e-07 | 1.4e-07 | 0.662 +- 0.002 | 0.696 | - |
| 128 | 1 | FEM | fem_lin | 0.0408 | 1.01e-06 | 6.31e-07 | 0.377 +- 0.008 | 0.428 | 0.25 |
| 128 | 1 | FEM | input | 0.0408 | 3.05e-05 | 1.17e-05 | 0.617 +- 0.009 | 0.682 | 0.97 |
| 128 | 1 | 4LPT matched | fem_lin | 0.0408 | 3.96e-07 | 1.52e-07 | 0.616 +- 0.014 | 0.646 | 0.25 |
| 128 | 1 | 4LPT raw | input | 0.0429 | 3.94e-07 | 1.42e-07 | 0.639 +- 0.018 | 0.665 | - |
| 128 | 1 | 2LPT matched | fem_lin | 0.0405 | 8.15e-07 | 3.27e-07 | 0.598 +- 0.019 | 0.592 | - |
| 128 | -1 | FEM | fem_lin | 0.0422 | 9.88e-07 | 6.37e-07 | 0.355 +- 0.004 | 0.426 | 0.15 |
| 128 | -1 | FEM | input | 0.0422 | 3.05e-05 | 1.17e-05 | 0.618 +- 0.005 | 0.690 | 0.95 |
| 128 | -1 | 4LPT matched | fem_lin | 0.0422 | 4.12e-07 | 1.62e-07 | 0.608 +- 0.009 | 0.613 | 0.13 |
| 128 | -1 | 4LPT raw | input | 0.0443 | 4.18e-07 | 1.56e-07 | 0.627 +- 0.012 | 0.622 | - |
| 128 | -1 | 2LPT matched | fem_lin | 0.0418 | 8.22e-07 | 3.37e-07 | 0.591 +- 0.012 | 0.581 | - |
| 128 | 0.5 | FEM | fem_lin | 0.00985 | 1.51e-07 | 1e-07 | 0.334 +- 0.008 | 0.395 | 0.33 |
| 128 | 0.5 | FEM | input | 0.00985 | 6.99e-06 | 2.75e-06 | 0.606 +- 0.007 | 0.672 | 0.99 |
| 128 | 0.5 | 4LPT matched | fem_lin | 0.00985 | 5.4e-08 | 2.19e-08 | 0.595 +- 0.008 | 0.647 | 0.59 |
| 128 | 0.5 | 4LPT raw | input | 0.0103 | 4.72e-08 | 1.6e-08 | 0.662 +- 0.006 | 0.698 | - |
| 128 | 0.5 | 2LPT matched | fem_lin | 0.00984 | 6.5e-08 | 2.41e-08 | 0.630 +- 0.007 | 0.669 | - |
| 128 | -0.5 | FEM | fem_lin | 0.01 | 1.47e-07 | 1e-07 | 0.322 +- 0.008 | 0.385 | 0.33 |
| 128 | -0.5 | FEM | input | 0.01 | 6.98e-06 | 2.75e-06 | 0.607 +- 0.004 | 0.677 | 0.99 |
| 128 | -0.5 | 4LPT matched | fem_lin | 0.01 | 5.37e-08 | 2.15e-08 | 0.600 +- 0.006 | 0.651 | 0.59 |
| 128 | -0.5 | 4LPT raw | input | 0.0105 | 4.73e-08 | 1.57e-08 | 0.667 +- 0.004 | 0.701 | - |
| 128 | -0.5 | 2LPT matched | fem_lin | 0.00999 | 6.47e-08 | 2.38e-08 | 0.633 +- 0.006 | 0.672 | - |


* With its own linear tensor (`fem_lin`) the FEM behaves like the matched LPT in the one-scale residual (V_C 3.5e-6 vs 1.9e-6 at N=64, eps=1; 1.0e-6 vs 4.0e-7 at N=128), but the parent explains less of it (R_gain 0.32-0.55 vs 0.60-0.69): the extra FEM residual (V_CP(FEM) - V_CP(4LPT matched) = 1.3e-6 at N=64, 4.8e-7 at N=128) is FEM-specific dynamics/discretisation not captured by (A_C, A_P).
* With the *input* linear field as the "initial" tensor the FEM residual variance is 30-90x larger (rms ln J residual 2.6e-3 ... 1.7e-2): this is the mesh-distorted linear response (a scale-dependent transfer function), and the parent absorbs a part of it (R_gain 0.6, flexible 0.95-0.99).  This is the caveat of SUMMARY.md Sec. 9, now measured at the cube level: the two-scale gain for FEM-with-input-A is a linear-response correction, not an environment effect.
* Single-seed spatial split: errors are indicative only.  Train R_gain exceeds test R_gain by 0.03-0.09 for FEM (slab-to-slab variation with ~1e5 cubes and 26 parameters); flexible FEM fits at N=128 (455 parameters, 9e5 cubes) are noisy.

## 5. CDF bookkeeping (`cdf_multiscale.csv`)

F_M = P(J < 1), F+- = P(delta_L < 0, J < 1), F-+ = P(delta_L > 0, J > 1), with delta_L = -eps tr A_C at the cube level, evaluated on all test cubes; "(one)"/"(two)" use predicted J from the one-/two-scale ln J fit (cut J > 0.1, invariants).  Quoted: actual value, then (prediction - actual) +- scatter among the 4 test groups.

Gaussian (test seeds 5-8, 8.39M cubes):

| method | R_f | eps | cube tensor | F_M actual | F_M(one)-actual | F_M(two)-actual | F+- actual | dF+-(one) | dF+-(two) | F-+ actual/one/two | P(dL>0) | sigma_cube |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4LPT | 14 | 1 | A_C | 0.52072 | +1.8e-04 +- 1e-05 | +1.4e-04 +- 2e-05 | 2.04e-02 | +1.2e-04 | +1.1e-04 | 5.71e-05 / 0.00e+00 / 2.41e-05 | 0.50036 | 0.2696 |
| 4LPT | 14 | 1 | A_K | 0.52072 | +1.5e-04 +- 3e-05 | +1.5e-04 +- 2e-05 | 2.03e-02 | +1.5e-04 | +1.5e-04 | 4.77e-07 / 0.00e+00 / 0.00e+00 | 0.50040 | 0.2669 |
| 4LPT | 18 | 1 | A_C | 0.51615 | +5.5e-05 +- 3e-05 | +1.9e-05 +- 3e-06 | 1.57e-02 | +2.9e-05 | +6.9e-06 | 2.57e-05 / 0.00e+00 / 1.32e-05 | 0.50047 | 0.2082 |
| 4LPT | 18 | 1 | A_K | 0.51615 | +2.5e-05 +- 9e-06 | +2.8e-05 +- 9e-06 | 1.57e-02 | +2.4e-05 | +2.8e-05 | 9.54e-07 / 0.00e+00 / 0.00e+00 | 0.50046 | 0.2068 |
| 4LPT | 18 | 0.5 | A_C | 0.50832 | +1.4e-05 +- 2e-05 | -5.7e-06 +- 1e-05 | 7.95e-03 | -8.2e-05 | -4.5e-05 | 9.62e-05 / 0.00e+00 / 5.71e-05 | 0.50047 | 0.1041 |
| 4LPT | 18 | 0.5 | A_K | 0.50832 | +5.2e-06 +- 6e-06 | +2.9e-06 +- 6e-06 | 7.86e-03 | +2.7e-06 | +3.6e-07 | 2.50e-06 / 0.00e+00 / 0.00e+00 | 0.50046 | 0.1034 |
| 4LPT | 22 | 1 | A_C | 0.51275 | +4.2e-05 +- 1e-05 | -4.1e-06 +- 7e-06 | 1.26e-02 | +2.7e-05 | -1.1e-05 | 1.50e-05 / 0.00e+00 / 7.99e-06 | 0.50014 | 0.1668 |
| 4LPT | 22 | 1 | A_K | 0.51275 | +1.1e-05 +- 1e-05 | +1.5e-05 +- 1e-05 | 1.26e-02 | +1.1e-05 | +1.5e-05 | 0.00e+00 / 0.00e+00 / 0.00e+00 | 0.50011 | 0.1660 |
| 2LPT | 14 | 1 | A_C | 0.52057 | +7.9e-05 +- 3e-05 | +3.6e-05 +- 3e-05 | 2.03e-02 | +1.5e-05 | +2.9e-06 | 6.34e-05 / 0.00e+00 / 3.06e-05 | 0.50036 | 0.2696 |
| 2LPT | 14 | 1 | A_K | 0.52057 | +5.0e-05 +- 4e-05 | +5.2e-05 +- 3e-05 | 2.02e-02 | +4.9e-05 | +5.1e-05 | 8.34e-07 / 0.00e+00 / 0.00e+00 | 0.50040 | 0.2669 |
| 2LPT | 18 | 1 | A_C | 0.51608 | +1.3e-05 +- 3e-05 | -2.3e-05 +- 4e-06 | 1.56e-02 | -1.6e-05 | -3.5e-05 | 2.83e-05 / 0.00e+00 / 1.56e-05 | 0.50047 | 0.2082 |
| 2LPT | 18 | 1 | A_K | 0.51608 | -1.8e-05 +- 7e-06 | -1.3e-05 +- 8e-06 | 1.56e-02 | -1.8e-05 | -1.4e-05 | 9.54e-07 / 0.00e+00 / 0.00e+00 | 0.50046 | 0.2068 |
| 2LPT | 18 | 0.5 | A_C | 0.50831 | +1.1e-05 +- 2e-05 | -7.6e-06 +- 9e-06 | 7.95e-03 | -8.9e-05 | -4.8e-05 | 9.98e-05 / 0.00e+00 / 5.94e-05 | 0.50047 | 0.1041 |
| 2LPT | 18 | 0.5 | A_K | 0.50831 | +3.2e-06 +- 3e-06 | +2.9e-06 +- 4e-06 | 7.85e-03 | +6.0e-07 | +2.4e-07 | 2.62e-06 / 0.00e+00 / 0.00e+00 | 0.50046 | 0.1034 |
| 2LPT | 22 | 1 | A_C | 0.51274 | +5.6e-06 +- 9e-06 | -3.9e-05 +- 1e-05 | 1.26e-02 | -1.0e-05 | -4.7e-05 | 1.61e-05 / 0.00e+00 / 8.23e-06 | 0.50014 | 0.1668 |
| 2LPT | 22 | 1 | A_K | 0.51274 | -2.3e-05 +- 2e-05 | -2.7e-05 +- 2e-05 | 1.26e-02 | -2.3e-05 | -2.7e-05 | 1.19e-07 / 0.00e+00 / 0.00e+00 | 0.50011 | 0.1660 |


FEM (single seed; test = 4 sub-slabs; cut J > 0.1; ln J, invariants):

| N | eps | target | A source | F_M actual | F_M(one)-actual | F_M(two)-actual | F+- actual/one/two | F-+ actual/one/two | sigma_cube |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 1 | FEM | fem_lin | 0.55235 | -2.4e-04 +- 3e-04 | -2.4e-04 +- 7e-05 | 1.44e-02 / 1.40e-02 / 1.41e-02 | 1.09e-04 / 0.00e+00 / 3.62e-05 | 0.1854 |
| 64 | 1 | FEM | input | 0.55235 | -8.7e-04 +- 3e-03 | +2.7e-04 +- 1e-03 | 2.51e-02 / 1.64e-02 / 2.29e-02 | 7.86e-03 / 0.00e+00 / 5.33e-03 | 0.2018 |
| 64 | 1 | 4LPT matched | fem_lin | 0.55215 | -1.7e-04 +- 4e-04 | -9.0e-05 +- 2e-04 | 1.41e-02 / 1.39e-02 / 1.40e-02 | 0.00e+00 / 0.00e+00 / 0.00e+00 | 0.1854 |
| 64 | -1 | FEM | fem_lin | 0.47669 | +1.4e-04 +- 3e-04 | -6.3e-05 +- 2e-04 | 1.48e-02 / 1.49e-02 / 1.47e-02 | 9.04e-05 / 9.04e-06 / 3.62e-05 | 0.1854 |
| 64 | -1 | FEM | input | 0.47669 | +2.3e-03 +- 3e-03 | +7.5e-04 +- 2e-03 | 2.28e-02 / 1.43e-02 / 2.00e-02 | 1.10e-02 / 2.53e-04 / 7.43e-03 | 0.2018 |
| 64 | -1 | 4LPT matched | fem_lin | 0.47663 | +7.2e-05 +- 2e-04 | -9.0e-06 +- 1e-04 | 1.47e-02 / 1.48e-02 / 1.47e-02 | 0.00e+00 / 0.00e+00 / 0.00e+00 | 0.1854 |
| 64 | 0.5 | FEM | fem_lin | 0.54527 | -2.3e-04 +- 9e-05 | +9.9e-05 +- 5e-05 | 7.34e-03 / 6.98e-03 / 7.36e-03 | 1.36e-04 / 0.00e+00 / 5.43e-05 | 0.0927 |
| 64 | 0.5 | FEM | input | 0.54527 | -1.6e-03 +- 3e-03 | -1.4e-04 +- 2e-03 | 2.07e-02 / 8.63e-03 / 1.75e-02 | 1.05e-02 / 0.00e+00 / 7.42e-03 | 0.1009 |
| 64 | 0.5 | 4LPT matched | fem_lin | 0.54534 | -3.0e-04 +- 2e-04 | -9.0e-06 +- 1e-04 | 7.33e-03 / 6.97e-03 / 7.29e-03 | 6.33e-05 / 0.00e+00 / 2.71e-05 | 0.0927 |
| 64 | -0.5 | FEM | fem_lin | 0.46913 | +3.1e-04 +- 2e-04 | +1.3e-04 +- 6e-05 | 7.32e-03 / 7.51e-03 / 7.41e-03 | 1.27e-04 / 0.00e+00 / 9.04e-05 | 0.0927 |
| 64 | -0.5 | FEM | input | 0.46913 | +2.0e-03 +- 3e-03 | +9.9e-04 +- 2e-03 | 1.76e-02 / 6.61e-03 / 1.48e-02 | 1.33e-02 / 3.53e-04 / 9.63e-03 | 0.1009 |
| 64 | -0.5 | 4LPT matched | fem_lin | 0.46927 | +9.0e-05 +- 5e-05 | -4.5e-05 +- 9e-05 | 7.41e-03 / 7.43e-03 / 7.32e-03 | 6.33e-05 / 0.00e+00 / 1.81e-05 | 0.0927 |
| 128 | 1 | FEM | fem_lin | 0.55193 | -1.3e-04 +- 1e-04 | -2.7e-05 +- 6e-05 | 1.62e-02 / 1.61e-02 / 1.62e-02 | 6.78e-06 / 0.00e+00 / 1.13e-06 | 0.1985 |
| 128 | 1 | FEM | input | 0.55193 | -7.9e-04 +- 8e-04 | -1.9e-04 +- 4e-04 | 1.77e-02 / 1.63e-02 / 1.72e-02 | 5.53e-04 / 0.00e+00 / 3.04e-04 | 0.2032 |
| 128 | 1 | 4LPT matched | fem_lin | 0.55190 | -7.7e-05 +- 1e-04 | +1.2e-05 +- 1e-04 | 1.62e-02 / 1.61e-02 / 1.62e-02 | 0.00e+00 / 0.00e+00 / 0.00e+00 | 0.1985 |
| 128 | -1 | FEM | fem_lin | 0.48038 | +1.2e-04 +- 1e-04 | +2.0e-04 +- 8e-05 | 1.61e-02 / 1.62e-02 / 1.63e-02 | 6.78e-06 / 0.00e+00 / 0.00e+00 | 0.1985 |
| 128 | -1 | FEM | input | 0.48038 | +5.6e-04 +- 8e-04 | +3.0e-04 +- 4e-04 | 1.63e-02 / 1.58e-02 / 1.61e-02 | 1.05e-03 / 3.39e-06 / 6.02e-04 | 0.2032 |
| 128 | -1 | 4LPT matched | fem_lin | 0.48038 | +1.0e-04 +- 1e-04 | +1.3e-04 +- 9e-05 | 1.61e-02 / 1.62e-02 / 1.62e-02 | 0.00e+00 / 0.00e+00 / 0.00e+00 | 0.1985 |
| 128 | 0.5 | FEM | fem_lin | 0.54384 | -3.5e-05 +- 7e-05 | +3.6e-05 +- 6e-05 | 8.15e-03 / 8.10e-03 / 8.18e-03 | 1.36e-05 / 0.00e+00 / 1.13e-06 | 0.0992 |
| 128 | 0.5 | FEM | input | 0.54384 | -7.5e-04 +- 7e-04 | -2.4e-04 +- 4e-04 | 1.03e-02 / 8.27e-03 / 9.58e-03 | 1.32e-03 / 0.00e+00 / 8.08e-04 | 0.1016 |
| 128 | 0.5 | 4LPT matched | fem_lin | 0.54386 | -4.5e-05 +- 5e-05 | +6.8e-06 +- 2e-05 | 8.16e-03 / 8.11e-03 / 8.16e-03 | 0.00e+00 / 0.00e+00 / 0.00e+00 | 0.0992 |
| 128 | -0.5 | FEM | fem_lin | 0.47223 | +1.1e-04 +- 6e-05 | +7.9e-05 +- 7e-05 | 7.94e-03 / 8.04e-03 / 8.00e-03 | 1.24e-05 / 0.00e+00 / 0.00e+00 | 0.0992 |
| 128 | -0.5 | FEM | input | 0.47223 | +5.7e-04 +- 7e-04 | +1.6e-04 +- 4e-04 | 9.15e-03 / 7.64e-03 / 8.54e-03 | 2.09e-03 / 1.47e-05 / 1.33e-03 | 0.1016 |
| 128 | -0.5 | 4LPT matched | fem_lin | 0.47230 | +5.8e-05 +- 5e-05 | +1.5e-05 +- 3e-05 | 8.00e-03 / 8.06e-03 / 8.01e-03 | 0.00e+00 / 0.00e+00 / 0.00e+00 | 0.0992 |


* F_M - P(dL > 0) at R_f = 18, eps = 1: 0.51615 - 0.50046 = 0.0157 = 0.0757 sigma_cube (A1 = 0.0760); F_M - 1/2 = 0.0207/0.2696 = 0.0768 sigma at R_f = 14.
* One-scale errors: +1.8e-4 (R_f = 14), +5.5e-5 (18), +4.2e-5 (22) with A_C; two-scale: +1.4e-4, +1.9e-5, -4e-6.  With the correct edge tensor A_K the one-scale errors are +1.5e-4, +2.5e-5, +1.1e-5, and the parent changes nothing (+1.5e-4, +2.8e-5, +1.5e-5).  The residual +1e-4 at R_f = 14 survives in all variants (most likely the polynomial truncation at sigma_cube = 0.27; not removed by the parent).  All of these are at most ~1 % of F_M - 1/2.
* F-+ is tiny but not zero in the actual cubes (5.7e-5, 2.6e-5, 1.5e-5 for R_f = 14, 18, 22 with delta_L from A_C; ~1e-6 or 0 when delta_L is taken from A_K): the A_C-based delta_L=0 surface is displaced by the quadrature offset.  The one-scale model predicts F-+ = 0, the two-scale model recovers about half of it (2.4e-5 vs 5.7e-5 at R_f = 14).  This is again the quadrature effect, and it is irrelevant at 1e-5.
* FEM with fem_lin: F_M(pred) - F_M(actual) = (0.3-3)e-4 for both scales, within or near the sub-slab scatter, for eps = +-0.5, +-1 and N = 64, 128; matched 4LPT likewise.  With the input A (mesh-distorted) the errors are (0.6-2.3)e-3 and F+- is badly off (up to 1e-2): the cube-level delta_L from the input field is the wrong linear field for the FEM.

## 6. Innovation response (`innovation_response.csv`, `fig_innovation.png`)

eps_J = ln J - f(A_C) (one-scale fit, test cubes with J > 0.1) was binned into tertiles of eps tr A_P and regressed (univariate and jointly) on |eta_0| (trace innovation), eta_E:eta_E and eta_T:eta_T, with eta = eps (A_C - beta A_P) sector-wise (beta from the Gaussian box prediction).

* Gaussian: all correlations are <= 0.026 in magnitude (joint R^2 <= 1e-3) in every tertile and R_f; no dependence of the response on the parent state is detectable.  The residual rms itself does depend on the parent: it is 35 % larger in the underdense-parent tertile than in the overdense one (R_f = 18, eps = 1: 2.1e-3, 1.7e-3, 1.55e-3), a heteroscedasticity of the (tiny) nonlinear quadrature residual, not a response to innovation.
* FEM with fem_lin at N = 128 (eps = 1): corr(|eta_0|) = +0.09 / -0.02 / -0.12 in the three tertiles (R^2 <= 1.6 %), i.e. at most a weak, sign-changing trace-innovation coupling in 3e5-cube bins; this is small and not a basis for a theory.

## 7. Sanity checks

* <J_cube> = 1 in every case: |<J> - 1| <= 5e-9 for all LPT runs (seed 1: 0.99999999x), FEM `meanJ` = 1.000000.
* A_P equals the 27-cube periodic average of A_C: max |A_P - avg27(A_C)| = 2.2e-16 ... 3.3e-16 (relative to |A| ~ 0.25) for all seeds; Nyquist power in the G1 components 2e-14; own FFT gradient of psi_1 reproduces DISCO-DJ's `evaluate_jacobian_from_psi` to 2.5e-7 (float32).
* Sector variances: measured vs box sum within 0.6 % (L = 1000), see Sec. 3.
* One-scale ln J fit versus the expansion ln J = I1 - I1^2/2 + (4/7) I2 + O(A^3) (from -ln J = delta + (13/42) delta^2 + (2/7) T:T with delta = -I1, T:T = (2/3) I1^2 - 2 I2), seed 1, R_f = 18, all cubes with J > 0:

| method | eps | full 11-term fit: c(I1), c(I1^2), c(I2) | 3-term fit (I1,I1^2,I2) | rms resid (11-term) | rms resid (fixed 2nd-order expansion) | rms ln J |
|---|---|---|---|---|---|---|
| 4LPT | 0.1 | 0.9933 / -0.4829 / 0.5400 | 0.9935 / -0.4902 / 0.5640 | 1.56e-04 | 2.06e-04 | 0.021 |
| 4LPT | 0.2 | 0.9933 / -0.4880 / 0.5516 | 0.9942 / -0.4920 / 0.5642 | 3.12e-04 | 3.94e-04 | 0.041 |
| 4LPT | 0.5 | 0.9932 / -0.4909 / 0.5583 | 0.9993 / -0.4962 / 0.5646 | 8.00e-04 | 8.93e-04 | 0.103 |
| 4LPT | 1 | 0.9917 / -0.4890 / 0.5585 | 1.0185 / -0.5103 / 0.5638 | 1.78e-03 | 5.52e-03 | 0.211 |
| 2LPT | 0.1 | 0.9933 / -0.4829 / 0.5398 | 0.9935 / -0.4901 / 0.5638 | 1.56e-04 | 2.06e-04 | 0.021 |
| 2LPT | 0.2 | 0.9933 / -0.4879 / 0.5512 | 0.9941 / -0.4919 / 0.5640 | 3.14e-04 | 3.98e-04 | 0.041 |
| 2LPT | 0.5 | 0.9932 / -0.4907 / 0.5573 | 0.9985 / -0.4952 / 0.5651 | 8.22e-04 | 8.90e-04 | 0.103 |
| 2LPT | 1 | 0.9921 / -0.4894 / 0.5559 | 1.0148 / -0.5059 / 0.5689 | 2.00e-03 | 4.50e-03 | 0.211 |


  The coefficients are within 1 % (I1: 0.993), 2-3 % (I1^2: -0.483 ... -0.489 vs -0.5) and 2-6 % (I2: 0.54 ... 0.56 vs 0.571) of the point-value expansion; the 0.7 % shortfall of c(I1) is exactly the A_C-vs-edge quadrature offset of Sec. 4.1 (projected on tr A_C), so "a cube-averaged A is not the point value" is quantitatively visible.  2LPT and 4LPT give the same coefficients to <1 % at eps <= 0.5.

## 8. Caveats

* h/R_f ranges from 0.18 to 1.1 in the predictions; the N = 128 (h/R_f = 0.36-0.56) cases that matter have f_new of 1.7e-3 ... 8e-3.  At h = 15.6 (N = 64) the correlation loses 2-9 % in variance, but no regressions were run at N = 64 for the Gaussian ensemble (FEM N=64 has h = 7.8).
* Cube vs vertex Jacobians: the vertex Jacobian of the production ensembles (FFT gradient, point values) differs from the Kuhn cube J by the quadrature effect above; cube-level F_M, P(dL > 0) and sigma_cube (0.208 at R_f = 18, vs 0.2108 vertex) are slightly different from the vertex-level numbers of SUMMARY.md.
* R_gain is a relative statistic of an already tiny residual; it should always be quoted together with V_C/Var (Sec. 4) and the CDF shifts (Sec. 5).  Its value of 0.65-0.69 is a property of the Kuhn-cube estimator, independent of R_f and eps.
* The FEM results use a single realization (spatial split), tiny test errors are indicative only; the FEM linear-response issue (mesh-distorted psi_lin) is why `fem_lin`/`matched` is the fair comparison.  4LPT/2LPT `matched` cases use `with_external_ics(delta = -tr G_lin)`, so their A_C is the gradient of psi_lin up to the symmetrisation.
* The flexible (cubic raw-component) model is only a cross-check; for ln J with a J > 0.1 training cut it extrapolates poorly to J < 0.1 cubes.
* No tuning: bases and cuts were fixed in advance; the A_K variant (Sec. 4.1) is an added diagnostic, not a replacement of the requested analysis, and uses the same A_P.
* Heavy machine load: all jobs ran at nice 10; no other process was touched.

## 9. Files

* Code: `multiscale/src/MS.jl`, `multiscale/src/Regress.jl`, `multiscale/bin/{_env,_csv,build_cubes,build_cubes_K,build_fem_cubes,scale_correlations,regress,merge,sanity,diag_quadrature,save_arrays,make_figs,test_small}.jl`.
* Tables: `scale_correlations.csv`, `locality_gain.csv`, `cdf_multiscale.csv`, `innovation_response.csv`, `oneScale_coefficients.csv`, `diag_quadrature.csv`, `sanity_lnJ_coefficients.csv`; per-case parts in `parts/`.
* Figures: `fig_fnew_vs_Rf.png`, `fig_residual_rms.png`, `fig_Rgain_vs_Rf.png`, `fig_innovation.png`.
* Per-cube arrays (JLD2, compressed; layout tensors (6,N,N,N) [comp,i,j,k] comps xx,yy,zz,xy,xz,yz, scalars (N,N,N)[i,j,k]; keys A_C, A_P, eta_C, delta_L_cube, J_<n>LPT_e<eps>, FEM_e<eps>): `data/multiscale/arrays_L1000_N128_R18_seed1.jld2`, `arrays_FEM_L500_N64_seed1.jld2`, `arrays_FEM_L500_N128_seed1.jld2`; raw per-seed caches `cubes_*.jld2`, `cubesK_*.jld2`.
