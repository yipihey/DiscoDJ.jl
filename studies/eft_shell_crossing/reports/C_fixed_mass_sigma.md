# Report C: variance at fixed mass, spherical collapse and the reach of the UV

Scripts: on branch `eft-lcdm-cs`, `twoloop/sigmaM.jl`, `sigmaM_analysis.py`, `sigmaM_sc.py`, `sc_lowmass.py`, `patch_anisotropy.py`, `patch_analysis.py`, `sigmaM_pair.py`, `pk_snap.jl`, `k_of_R.py`, `k_of_R_nowiggle.py`; on branch `fixed-mass-sheet-moments`, `src/fixedmass.jl`, `scripts/run_fixedmass.jl`, `fixedmass_analysis.py`, `fixedmass_response.py`, `figures_fixedmass.py`. Figures: `figM1`–`figM5`, `figU1`, `figF1`.

## 1. Definitions

M is labelled by R₀ with M = ρ̄V(R₀), Ω_m = 1 (about 3.2× the mass at the same radius for Ω_m = 0.31). 5 × 10¹⁵ h⁻¹M☉ is R₀ = 16.3 h⁻¹Mpc.

| statistic | what is held fixed | centres |
|---|---|---|
| σ(δ_R) | Eulerian radius | uniform |
| random-centred σ(ln V) | mass | uniform |
| mass-drawn σ(ln V) | mass | drawn from ρ |
| Lagrangian patch σ(ln V) | the same mass elements | every mass element |

By P(R_M ≤ R) = P(M_R ≥ M), one weighted table F_m(R, μ) gives the volume distribution at fixed mass; it is the data-centred version of the kNN-CDF. Around points drawn from the density, the mean of δ_R is ξ̄(R), linear in P(k), while the variance is σ²(R) + ⟨δ δ_R²⟩ − ξ̄²(R), which includes an integrated bispectrum even at leading order.

All statistics agree with linear σ at D = 0.1 to about 1 % for 6 × 10¹⁴ ≲ M ≲ 3 × 10¹⁶ h⁻¹M☉, and to about 1.5 % at 2.5 × 10¹⁴. Patch σ uses signed volumes, which count only the enclosing surface; the unsigned sheet volume behaves very differently (0.379, 26 % below linear, at 3.7 × 10¹⁵, D = 1).

## 2. σ(M) (figM1)

Relative to linear σ at the same mass, D = 1, tl_full:

| M (h⁻¹M☉) | fixed radius | random centres | Lagrangian patch | mass-drawn |
|---|---|---|---|---|
| 3.7 × 10¹⁵ | +2.2 % | −5.7 % | +19 % | +35 % |
| 1.0 × 10¹⁶ | +0.5 % | −2.9 % | +7.0 % | +24 % |
| 1.5 × 10¹⁷ | −0.5 % | −0.5 % | +0.2 % | +3.3 % |

Mass-drawn σ departs earliest and most. At R₀ = 10, D = 1, on the 64³ sub-lattice of the fiducial pair, it separates PM from 3LPT by 31 %, against 4 % for random centres.

Reweighting centres by the sphere's own μ_R rather than ρ(x) closes 55–65 % of the mass-drawn minus patch gap (0.690, 0.636, 0.607 at 3.7 × 10¹⁵). So most of it is centre density inside the sphere, and the rest is the difference between Eulerian spheres and Lagrangian patches. That split assumes this reweighting model.

## 3. Spherical collapse (figM2, figM4)

Exact EdS collapse, frozen at Δ = 178, Gaussian δ_L at the Lagrangian radius, no fitted parameter other than the freeze.

* **Fixed radius needs contraction,** R_L = R μ^{1/3}. At D = 1 in tl_full:

  | M | measured σ(δ_R) | with contraction | without |
  |---|---|---|---|
  | 1.4 × 10¹⁵ | 0.703 | 0.719 | 1.141 |
  | 3.7 × 10¹⁵ | 0.524 | 0.534 | 0.731 |
  | 1.0 × 10¹⁶ | 0.385 | 0.392 | 0.447 |

  So σ(δ_R) ≈ linear is a cancellation in this model: without contraction it would be up to 71 % above linear at these masses. σ(ln μ_R) confirms it (0.480, against 0.485 with and 0.451 without, at 3.7 × 10¹⁵).
* **The non-Gaussianity of ln V at fixed mass comes almost entirely from mapping Gaussian δ_L through collapse.** Distributions agree over 2–3 decades of probability at M ≳ 10¹⁵ (figM4). Core widths, (q₈₄ − q₁₆)/2, unaffected by clipped patches:

  | M | linear | measured width | SC width |
  |---|---|---|---|
  | 2.5 × 10¹⁴ | 0.979 | 1.263 | 1.120 |
  | 9.7 × 10¹⁴ | 0.724 | 0.800 | 0.773 |
  | 3.8 × 10¹⁵ | 0.512 | 0.528 | 0.526 |

  Below about 5 × 10¹⁴ it explains about half of the excess. Past collapse, SC is a frozen spike and real patches a smooth tail. The freeze leaves core widths unchanged but moves the low-mass std (1.34, 1.54, 1.82 for Δ = 50, 178, 1000 at 2.5 × 10¹⁴).
* **It works in the UV-divergent spectrum as well,** with no retuning, though less tightly at fixed radius (3–5 % high at M ≤ 3.7 × 10¹⁵): 0.520 against 0.537 (0.737 without contraction) at 3.7 × 10¹⁵. At 2.5 × 10¹⁴ the core width is 1.235 against 1.208, closer than in the Planck-shape spectrum.
* **Random centres fit worst** (4–8 % low at D = 1 for M ≤ 10¹⁶, 1–2 % above). **Mass-drawn spheres have no SC prediction distinct from patches,** and exceed them.

## 4. Anisotropy (figM3)

| M | D | share of Var(ln V) not set by δ_L | scatter at fixed δ_L | a₃/a₁ of the image |
|---|---|---|---|---|
| 1.3 × 10¹⁵ | 1 | 13.8 % | 0.359 | 0.47 |
| 1.0 × 10¹⁶ | 1 | 4.3 % | 0.085 | 0.66 |
| 1.5 × 10¹⁷ | 1 | 2.7 % | 0.025 | 0.85 |

Images of low-mass patches are far from spherical, yet bins in δ_L alone account for most of Var(ln V): 86–97 % at M ≥ 1.3 × 10¹⁵ (71 % at 2.5 × 10¹⁴, in the divergent spectrum, D = 1). This is variance explained, not a causal decomposition. At fixed δ_L, higher initial shear means smaller volume (correlation −0.1 to −0.3), the sign the Zel'dovich product predicts, but the patch's own shear explains only 1–5 % of that scatter. Where the remainder comes from is not measured.

Unsmoothed, signed cell volumes vary far more: σ(ln|J|) = 0.85, 1.75, 2.28, 2.41 at D = 0.3, 0.5, 0.8, 1, with 0.5, 8.6, 23.9, 30.6 % of cells inverted. A patch's signed volume depends only on the motion of its surface, so interior folds cancel.

## 5. The UV and IR variance (figU1)

Seed-matched Var_A/Var_B − 1 minus the extra modes' linear power, D = 1:

| change | R = 20 | R = 30 | R = 50 | R = 78 | at R = 20, D = 0.3, 0.5, 0.8, 1 |
|---|---|---|---|---|---|
| k > 0.2 | −3.4 % | −2.4 % | −1.0 % | −0.6 % | −0.21, −0.71, −2.3, −3.4 |
| raise power above k = 0.3 | −1.29 % | −0.71 % | −0.31 % | −0.17 % | −0.41, −0.87, −1.22, −1.29 |
| k = 0.6–1.6 | −0.58 % | −0.32 % | −0.14 % | −0.07 % | −0.11, −0.34, −0.54, −0.58 |
| divergent, k = 1.0–1.6 | −0.37 % | −0.20 % | −0.09 % | −0.05 % | −0.25, −0.40, −0.41, −0.37 |
| k = 1.0–1.6 | −0.14 % | −0.08 % | −0.03 % | −0.02 % | −0.04, −0.10, −0.14, −0.14 |

* Suppression at every R ≥ 10 for all pairs. At R ≤ 6.9, the k > 0.2 row is positive at some epochs (up to +13.7 % at R = 5), where the subtraction is large and the added power's nonlinear growth matters.
* Spherical collapse misses this. It gives about 0 at R ≥ 20 for the deep-UV rows, and +4.2 % against −6.7 % at R = 15 for k > 0.2. At R ≤ 10 contraction moves SC about 70 % of the way, at R ≥ 20 almost none. It lacks the smearing of long modes by short displacements, which is a likely cause (not tested).
* **In the deep UV the fractional response saturates by D ≈ 0.8,** and the deepest divergent band turns over. The absolute variance difference still grows (+68 % from D = 0.8 to 1 for k = 0.6–1.6), since the variance itself grows. All modes above k = 0.2 keep growing fractionally.
* **The R dependence of the deep-UV rows is nearly universal,** about 1 : 0.55 : 0.23 : 0.13 across R = 20, 30, 50, 78. The amplitude per unit of I is not: −0.06 to −0.20 % per unit, a factor of 3.4, and the same k = 1.0–1.6 band gives half the response per I in the divergent spectrum.
* The same comparison at fixed mass agrees across statistics at R₀ ≥ 29 (to within about 0.07 percentage points in σ for the far-UV bands). At smaller R₀ mass-drawn σ is noisier and is not used.

**Separately, field-level on the one fiducial pair** at R = 4.48 and 6.69 (k_uv = 1.2, interpolated to R = 5): the coherent cross term 2⟨Δδ_IR⟩/⟨δ_IR²⟩ is −0.4, −2.8, −10, −24 % at D = 0.3, 0.5, 0.7, 1 for PM, and −51 % for 3LPT at D = 1. That is a different number from the variance response: ⟨Δ²⟩ itself grows above linear (1.42× at R = 4.48, D = 1), offsetting part of it, so total minus linear added power is −7.6 % at R = 5 in tl_full.

## 6. Converting k to R and M (figM5)

Setting Δ²(k, D) = σ²(R, D), linear theory gives one curve independent of D. For the smoothed spectrum kR = 1.37–1.46 at R = 6–60, rising to 1.57 at larger R. Nonlinearity moves k at fixed R by a median of 0.6 % by D = 1, up to 7 % near M ≈ 1.2 × 10¹⁵ where nonlinear evolution fills in a BAO dip, and ≤ 1.4 % above 10¹⁶. BAO adds −0.04 to +0.05: dips at M ≈ 1 × 10¹⁵ and 5 × 10¹⁵ fall on the P(k) wiggle maxima (k = 0.138, 0.078, period 0.061 h Mpc⁻¹), and vanish when the spectrum is smoothed. For quick use, M ≈ 3.6 × 10¹² k⁻³ h⁻¹M☉ for k in h Mpc⁻¹ (range 3.0–4.3 × 10¹²). Below M ≈ 2 × 10¹⁴ an offset of about 2–3 % appears at all D, probably because P(k) has the CIC window deconvolved and σ(R) does not.

## 7. Fixed-mass statistics of the fiducial pair (figF1)

Sheet statistics of one UV/IR pair, 3LPT, 7LPT (to D = 0.7) and PM, 10 epochs, 40 radii. No error bars except spatial scatter.

* **Mass-budget checks:** mean μ = 1 and node counts hold exactly. d⟨δ²⟩/dD closes to ≤ 1.5 % at R ≥ 10 for PM and 3LPT at D ≤ 0.5. The second-order budget closes to about 2 % for 3LPT at D ≤ 0.4 and R ≥ 10 (5 % at R = 5, D = 0.3); for PM to ≤ 1.2 % at R ≥ 22, and up to 19 % at R ≤ 10. PM's a is rebuilt from sheet-deposited gravity rather than its own force, which may be the cause (not tested). Instantaneous ∂_D F agrees with differences at μ = 1, and within about 10 % at μ = 0.5.
* **F(R, μ):** at R = 5, D = 1, P(μ ≥ 8) is 0.0059 in PM and 0.0011 in 3LPT: LPT cannot keep bound structures together.
* **Conditional budget:** at R = 10, D = 1, μ = 8–32, stress (−98.3) and acceleration (−98.1) nearly cancel, resembling a virial state, and dispersion is 55 % of the stress. At R = 5 they do not (−144, −92). One pair, one radius: a lead, no more.

## Limits

Fixed amplitudes and one box, so errors exclude cosmic variance. Radius grid starts at 4 h⁻¹Mpc, so mass-drawn σ below about 3 × 10¹⁵ at D = 1 is unresolved. Clipped patches inflate std at low M, so quantile widths are preferred there. Patches are sampled at 64³, overlapping at large M.
