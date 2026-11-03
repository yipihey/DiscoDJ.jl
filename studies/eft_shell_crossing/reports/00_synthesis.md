# Short modes, long modes and shell crossing: synthesis

Three linked measurements on EdS N-body and LPT runs (Planck18 spectral shape, σ₈ = 0.81 at D = 1, L = 500 h⁻¹Mpc, 256³ particles, 512³ force mesh, discreteness-suppressed PM; 200 BullFrog steps from D = 0.02, except report A's plane waves, 100 steps from D = 0.15 with 7LPT ICs):

| report | question | data |
|---|---|---|
| [A](A_density_response.md) | How do short modes change a long density mode? | ε → 0 plane waves on bands of short modes, 7 cutoffs |
| [B](B_two_loop_stress.md) | What effective stress do short modes exert, and does it match two-loop EFT? | 9 spectra and IR cuts, 4 seeds × ±L |
| [C](C_fixed_mass_sigma.md) | What does the variance do at fixed mass, and how far does the UV reach? | the same runs, plus one fixed-mass UV/IR pair |
| [D](D_campaign_notes.md) | Settings, machine behaviour, what went wrong | |

## Main results

1. **Short modes suppress long-wavelength density.** At k = 0.05, D = 1, the modes 0.2 ≤ k < 1.2 change P(k) by −0.97 % (C₂,δ = 1.935 ± 0.015 h⁻²Mpc²). Displacement responds with the opposite sign. An earlier version of the study published +0.2 % by applying the displacement coefficient to P(k); that is corrected. Density matches the one-loop P13 kernel to ≤ 1.2 % at all 7 cutoffs once three effects are removed (report A).

2. **The linear stress coefficient matches the UV prediction in far-UV bands of Planck-shape spectra, and not in the divergent one.** At D = 0.1, PM's stress c̄/I is 0.86–0.87 for k = 0.6–1.6 and 0.88–0.89 for k = 1.0–1.6, against the analytic 61/70 = 0.871. In the divergent spectrum's k = 1.0–1.6 band it is 1.04–1.05, 20 % high, and 7LPT agrees, so that excess is already in single-stream dynamics (report B).

3. **A stress that is local in time connects the channels best in Planck-shape spectra.** The measured c(D), through the EdS Green's function, gives the density and momentum responses to ≤ 4.4 % to D = 0.5 at Λ = 0.15 (≤ 8.5 % at Λ = 0.1) for those spectra. The divergent spectrum misses by 13–17 % from D ≈ 0.3 (k < 1.6), and by 3–11 % with k < 1.0 (report B).

4. **The eight-coefficient question stays open.** Only the linear combination is determined, with c_rδ,1 roughly measured. δ,2, δ²,1, δ,3, rδ,2 and rδ²,1 are not determined by this estimator. It is a limit of the method and more seeds would not fix it (report B).

5. **Spherical collapse describes the variance at fixed scale, and not the UV response.** With density-dependent contraction it reproduces σ(δ_R) to ≤ 2.3 % in the Planck-shape spectrum (≤ 5 % in the divergent one) and the patch ln V distribution over 2–3 decades of probability at M ≳ 10¹⁵ h⁻¹M☉. In this model, fixed-radius σ stays near linear because growth and contraction cancel. Patch images become elongated (a₃/a₁ = 0.47–0.66 at M ≤ 10¹⁶), yet bins in δ_L account for 86–97 % of Var(ln V) at M ≥ 1.3 × 10¹⁵ (report C).

6. **UV modes suppress IR variance weakly, and in the deep UV that fraction saturates.** At D = 1, modes at k = 0.6–1.6 change the variance by −0.58 % at R = 20 and −0.07 % at R = 78 h⁻¹Mpc. The *fractional* change stops growing by D ≈ 0.8, and in the deepest divergent band it turns over, though the absolute variance difference still grows. All modes above k = 0.2 (k < 1.6) keep growing fractionally. Spherical collapse misses this response: about 0 at R ≥ 20 for the deep UV, and the wrong sign for k > 0.2 at R = 15 (report C).

## How they fit together

Before crossing, PM and LPT agree in every statistic, and the spectra with more small-scale power cross earlier (the divergent spectrum's stress already differs between PM and 7LPT at D = 0.2). A coherent picture is that modes stop responding to the long field as they collapse. Several measurements on these same runs point that way: the far-UV stress coefficient falls (D ≈ 0.3–0.5, from a fit explaining only 11 % of the variance by D = 0.5), the fractional IR variance response flattens later (D ≈ 0.8), and time-local closure is worst in the spectrum that crossed earliest. The different timing, and the fact that k > 0.2 keeps growing, mean this is an interpretation, not a demonstrated mechanism. What is measured directly is that **no fixed power of D describes the response.**

One tension remains. Plane-wave density shells show no decoupling at ε → 0 (report A: every shell stays 1.16–1.58× one loop at D = 1), while the variance response and stress do saturate. These are different quantities, and they have not been reconciled.

## Limits common to all reports

* One box size, one set of phases per seed, EdS throughout. Masses use Ω_m = 1, about 3.2× the mass at the same radius for Ω_m = 0.31, so radius is the safer label.
* Fixed amplitudes and seed-matched differences give very small errors. They do not include cosmic variance.
* PM's own deposit bias is assumed equal to 7LPT's where it matters (report A).
* "Planck-shape" spectra here are EdS with the Planck18 shape. True ΛCDM enters only through the earlier c_s run on the same branch.
