# 2LPT vs N-body: smoothed density PDFs before shell crossing

**Question.** How well does 2LPT reproduce the *volume-weighted* and
*mass-weighted* PDFs of the smoothed density field at z = 0, on scales where
the flow has (almost) not shell-crossed, compared with a converged N-body
simulation from identical initial conditions?

**Short answer.** The 2LPT PDFs have converged to 0.01% and the N-body PDFs to
≲0.5%. Against that, 2LPT is **too narrow in both tails**. It has too few deep
voids and too few high-density regions:

| R_s [Mpc/h] | σ²₂LPT/σ²NB | S₃ 2LPT / NB | vol. 0.1% / 99.9% quantile | mass 99% / 99.9% quantile | central 10–90% |
|---:|---:|---:|---:|---:|---:|
|  7 (½ R_f) | 0.906 | 5.86 / 7.23 | +10.8% / −10.8% | −17.4% / −28.9% | ≤ 1.3% |
| 14 (R_f)   | 0.950 | 4.09 / 4.40 | +8.3% / −6.9%  | −7.5% / −9.1%   | ≤ 0.8% |
| 28 (2 R_f) | 0.972 | 2.86 / 2.94 | +2.8% / −1.7%  | −1.9% / −2.3%   | ≤ 0.4% |
| 42 (3 R_f) | 0.981 | 1.98 / 1.98 | +1.3% / −0.5%  | −0.5% / −0.5%   | ≤ 0.2% |

(Quantile columns: 2LPT/N-body − 1 of the density at that quantile. A positive
low quantile means the 2LPT voids are not empty enough, and a negative high
quantile means the 2LPT peaks are not dense enough.) The median agrees to ≤ 0.1%
on all scales. The paired-phase realisation reproduces every number to within
about 1 percentage point, so none of this is a feature of one phase draw. The
2LPT–N-body differences are typically 10–100× the numerical uncertainty of
either method.

## Setup

| | |
|---|---|
| Cosmology | Planck18 (EE+BAO+SN), the DiscoDJNative default; Eisenstein–Hu (1998) transfer with baryons, σ₈ = 0.8105 |
| Box | L = 300 Mpc/h, periodic |
| IC filter | **real-space spherical top hat**, R_f = 14 Mpc/h, applied to δ_lin: δ̂ → W_TH(kR_f) δ̂. Mass scale M_f = (4π/3) ρ̄ R_f³ = 9.8 × 10¹⁴ M☉/h. σ_lin(R_f, z=0) = 0.535 |
| Amplitudes | **fixed** (Angulo & Pontzen 2016): \|δ̂_k\| = N³ √(P(k)/V) exactly, random phases. A second **paired** realisation (θ → θ + π) is used as a robustness check |
| Phases | drawn once on a 512³ master grid; every resolution uses the sub-cube of the same modes, so all runs share identical phases for every mode they represent |
| Smoothing | real-space top hat, R_s = 7, 14, 28, 42 Mpc/h (½, 1, 2, 3 R_f) |
| Density estimator | CIC deposit onto a common 256³ analysis mesh for every run, CIC window deconvolved, then top-hat smoothed in Fourier space |
| Volume-weighted PDF | every analysis-mesh cell counts equally |
| Mass-weighted PDF | each cell weighted by its CIC-deposited particle mass, i.e. the ρ_R seen by a random mass element; independent of particle number |

### Choosing R_f: < 1% shell-crossed mass

Shell crossing is measured directly. Each Lagrangian lattice cube is split into
6 tetrahedra, and a tetrahedron counts as shell-crossed once its signed volume
has been ≤ 0 at any time. For N-body this is checked after every step; for
2LPT it is checked on 37 epochs along the 2LPT trajectory.

Pilot runs at 128³ gave an N-body ever-crossed mass fraction of 1.4%, 0.53% and
0.20% for R_f = 12, 14 and 16 Mpc/h. The Zel'dovich (Doroshkevich) estimate
badly underpredicts this; at σ = 0.6 it gives only 0.2%. R_f = 14 Mpc/h was
adopted. At full resolution:

| | ever shell-crossed mass | at z = 0 |
|---|---:|---:|
| N-body 256³ (fixed) | 0.52% | 0.45% |
| N-body 256³ (paired) | 0.69% | 0.56% |
| 2LPT 512³ (fixed / paired) | 0.19% / 0.26% | 0.19% / 0.26% |

The N-body fraction still creeps up slowly with resolution (0.23, 0.35, 0.42,
0.49, 0.52% for 64³–256³, with increments shrinking by about 2× per step) and
with force resolution. Extrapolated, it lands around 0.6% (fixed) and 0.8%
(paired), still below 1%. 2LPT crosses 2.5–3× less mass than N-body because
2LPT delays pancake collapse (`figures/shell_crossing.png`).

## Methods

All code is in this directory (numpy + scipy.fft + numba). Julia could not be
installed in the environment this study ran in, and DiscoDJNative has no N-body
solver yet. The pieces mirror the DiscoDJNative conventions:
`x = q + D₁ψ₁ + D₂ψ₂`, with D₁(1) = 1 and D₂ → −3/7 D₁².

* `cosmo.py`: E(a), EH98 P(k) (a line-by-line port of
  `DiscoDJNative.eisenstein_hu`), and the **exact** ΛCDM D₁ and D₂ from their
  ODEs. D₂(1)/D₁² = −0.4322, compared with −3/7 Ω_m^(−1/143) = −0.4321. Also
  f₁, f₂ and the leapfrog kick/drift integrals.
* `core.py`
  * ICs: fixed-amplitude, top-hat-filtered δ̂; a check gives σ(R_f) = 0.6032
    measured vs 0.6042 theory on the pilot grid.
  * 2LPT: ψ₁ = −∇φ₁ with ∇²φ₁ = δ, and ψ₂ = ∇φ₂ with
    ∇²φ₂ = Σ_{i>j}(φ,ᵢᵢφ,ⱼⱼ − φ,ᵢⱼ²). Velocities use f₁ and f₂.
  * PM N-body: KDK leapfrog in a with exact kick/drift integrals, steps uniform
    in ln a. CIC deposit and interpolation, Fourier Poisson solve, and a
    **4-point finite-difference gradient kernel** on a mesh of 2 × N_part.
  * Tetrahedral shell-crossing flags, and the smoothed-field PDFs.
* `run.py` and `queue.sh` run the whole study (22 runs, about 5 h on 4 cores).
  `analyze.py` produces `results/summary.json` and `figures/`.

**A PM pitfall found during validation.** With the force mesh at 2 × N_part,
the unperturbed particle lattice deposits a density pattern exactly at the
mesh Nyquist frequency. The spectral gradient (ik) turns this into O(1)
spurious lattice forces: a 50–100% error in the force at the filter scale.
Deconvolving the CIC window is worse still, because the lattice becomes
dynamically unstable. The fd4 kernel vanishes at Nyquist, and without
deconvolution it gives a force error of 1% at k < 0.05 h/Mpc even for 64³.
It also reproduces linear growth D₁(a) to 1–3% from z = 24 with 40 steps at 64³.
That configuration is used throughout. The remaining error is lattice
discreteness, and the resolution study shows it converging away.

## Resolution study

Metric: the largest relative shift of the 1%, 10%, 50%, 90% and 99% density
quantiles against the reference run (`figures/conv_*.png`,
`summary.json → convergence`).

**2LPT** (reference 512³; particle grid = 2LPT grid, 64³–512³):

| N | R_s = 7 (V / M) | 14 | 28 | 42 |
|---:|---:|---:|---:|---:|
| 64  | 10.6% / 7.9% | 0.63% / 0.60% | 0.03% | 0.01% |
| 128 | 0.55% / 1.4% | 0.03% | < 0.01% | < 0.01% |
| 256 | 0.02% / 0.03% | < 0.01% | < 0.01% | < 0.01% |
| 384 | < 0.01% | < 0.01% | < 0.01% | < 0.01% |

**N-body** (reference 256³ particles, 512³ PM mesh, 100 steps, z_i = 24):

| N | R_s = 7 (V / M) | 14 (V / M) | 28 | 42 |
|---:|---:|---:|---:|---:|
| 64  | 12.1% / 10.1% | 1.7% / 1.6% | 0.39% | 0.10% |
| 96  | 2.3% / 2.8% | 0.42% / 0.53% | 0.10% | 0.04% |
| 128 | 0.65% / 1.75% | 0.13% / 0.27% | 0.04% | 0.02% |
| 192 | 0.09% / 0.36% | 0.02% / 0.07% | 0.01% | < 0.01% |

N-body numerical parameters (shift against the fiducial run at 128³ or 256³, R_s = 7, V / M):

| test | quantile shift | σ² shift |
|---|---:|---:|
| 200 vs 100 steps (128³) | 0.03% / 0.05% | −0.07% |
| 50 vs 100 steps (256³) | 0.11% / 0.29% | +0.27% |
| 25 vs 100 steps (128³) | 0.56% / 1.2% | +1.4% |
| force mesh 3× vs 2× (128³) | 0.26% / 0.51% | +0.17% |
| force mesh 1× vs 2× (128³) | 3.0% / 2.5% | −2.9% |
| z_i = 49 vs 24 | 0.44% / 0.33% | −0.04% |
| z_i = 11 vs 24 | 0.22% / 0.17% | −0.13% |

Both methods converge monotonically to their own answers. At the reference
resolutions the residual uncertainty is ≤ 0.01% for 2LPT and ≲ 0.5% for N-body
on every scale, even at R_s = ½R_f. These two are the shaded bands in
`figures/compare_*.png` and the error bars in `figures/compare_quantiles.png`.
Both are invisible next to the 2LPT–N-body differences. The σ², S₃ and S₄
curves in `figures/moments_vs_resolution.png` are flat to plotting accuracy
from 128³ up for 2LPT and from 192³ up for N-body.

## Results: 2LPT vs N-body

Figures: `compare_vol.png`, `compare_mass.png`, `compare_quantiles.png`,
`field_level.png`. Full numbers are in `results/summary.json → comparison`.

1. **The bulk is excellent, the tails are not.** The median agrees to ≤ 0.1%,
   and the 10–90% range to ≲ 1% on all scales. The differences are all in the
   tails and grow as R_s decreases.
2. **2LPT's voids are too dense.** The 0.1% (1%) volume quantile of 1+δ is
   10.8% (5.6%) too high at R_s = 7 and 8.3% (4.0%) too high at R_s = R_f.
3. **2LPT's peaks are not dense enough, and mass weighting amplifies this.**
   The volume-weighted 99.9% quantile is 11% low at R_s = 7 and 7% low at 14.
   The mass-weighted 99% and 99.9% quantiles are 17% and 29% low at R_s = 7,
   and 7.5% and 9% low at R_s = R_f. The mass-weighted PDF puts more weight on
   the collapsing regions where 2LPT truncation matters most. Those regions
   include the ≲ 0.5% of mass that has shell-crossed in the N-body run.
4. **Moments.** σ²₂LPT/σ²NB = 0.91, 0.95, 0.97 and 0.98 at R_s = 7–42. The 2%
   deficit even at 3R_f is the missing 3LPT+ contribution to the power spectrum
   at k ≈ 0.05–0.1 h/Mpc. S₃ is 5–20% low for R_s ≤ R_f and within 1–3% for
   R_s ≥ 2R_f. S₄ is 42% low at R_s = 7, 19% low at R_f and 12% low at 2R_f.
   2LPT gets S₃ right at tree level but not the one-loop corrections, and not
   S₄ even at tree level.
5. **Field level.** Point-by-point, ln ρ₂LPT − ln ρNB has an rms of 1.7%, 1.2%,
   0.54% and 0.27% at R_s = 7–42. The correlation coefficient is ≥ 0.996. The
   PDF differences are therefore systematic, not scatter. The KS distance
   between the full 2LPT and N-body distributions is only 0.2–0.8%, which is
   why KS-type statistics say "agrees" while the tail quantiles do not.
6. **Robustness.** The paired-phase realisation changes the 2LPT–N-body
   quantile differences by ≤ 1 percentage point and the σ² ratio by ≤ 0.008.

**Fixed amplitudes and odd moments.** Fixed amplitudes remove the variance of
σ², but odd moments still depend on the phases. At R_s = 42 the fixed and
paired realisations give S₃ ≈ 1.98 and 3.2 for *both* methods, straddling the
tree-level value of 2.76. The 2LPT-vs-N-body comparison is unaffected because
both methods start from identical ICs. Absolute comparisons of S₃ or S₄ with
perturbation theory need the pair average, or more realisations.

## Caveats and possible extensions

* One box size (300 Mpc/h ≈ 21 R_f) and one fixed/paired phase pair. The
  largest smoothing scale (42 Mpc/h) contains only about 90 independent spheres,
  so its PDF shape is realisation-specific. The *ratios* between methods are
  not, as the paired check shows.
* The N-body is PM only, with a force resolution of 0.59 Mpc/h (R_f/24) at the
  top resolution. This is sufficient here, as the 3× mesh test shows. The
  shell-crossed fraction is the quantity most sensitive to force resolution.
* Natural next steps: add 3LPT (already in DiscoDJNative) to see how much of
  the tail and σ² deficit it recovers; a second R_f to map how the discrepancy
  scales with σ_lin; the phase-space-sheet (tetrahedral) density estimator from
  `DiscoDJNative/src/field/sheet_deposit.jl`.

## Reproduce

```bash
pip install numpy scipy numba matplotlib
cd studies/lpt_pdf
./queue.sh                  # ~5 h on 4 cores; fits in 15 GB RAM (2LPT 512³ is the peak)
LPTPDF_RESULTS=results LPTPDF_FIGURES=figures python3 analyze.py
```

Run outputs are staged in `_scratch/` (git-ignored); `results/` and `figures/`
hold the committed copies.
