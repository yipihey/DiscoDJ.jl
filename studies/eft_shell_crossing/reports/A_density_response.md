# Report A: the Eulerian density response to short modes

Full derivation and tables: main `README.md` §14.11. Scripts: `scripts/pw_density.jl`, `pw_density_exact.jl`, `pw_waves_only.jl`, `pw_density_analysis.py`, `figures_density.py`. Results: `results/pw_density*.json`. Figure: `figures/figP8_density_response.png`.

## Setup

Long field: ±ε Σ cos(k_j·x), ε = 0.01, 12 waves per run. Band: k_split ≤ |k| < 1.2 of each realization's white noise, 7 cutoffs, 4 realizations, 100 BullFrog steps from D = 0.15 with 7LPT ICs. Fits are R = −C₂,δk² on waves with k ≤ 0.1, so C₂,δ > 0 is suppression and ΔP/P = 2R. Errors are jackknife over realizations.

## Three corrections are needed

| effect | size | how it is removed |
|---|---|---|
| CIC deposit bias on near-lattice particles | +0.06 to +0.15 in C₂ until particles move about a cell; at k_split = 0.7 still +0.028 at D = 1 | direct sum of 7LPT at the waves |
| Finite ε: the waves smear each other; odd in ε, so ± keeps it | 0.02–0.04 D² | direct sum of the waves alone |
| Wave × band coupling at k_split ≤ 0.15 (k_j ∓ k_m in the band) | R ∝ D, realization noise | drop 2 waves per split |

The finite-ε term passes a falsification test: on the 6-wave ε = 0.01 and 0.02 runs the predicted change in 1LPT C₂ is +0.0063, +0.0124, +0.0253 at D = 0.5, 0.7, 1, against +0.0068, +0.0128, +0.0255 measured. It needs each wave's own Bessel term, −ε²D²/8, which a first estimate had omitted.

## Validation

Exact 3LPT with the waves removed, extrapolated to D → 0, against the standard P13 kernel with no free parameters:

| k_split | 0.12 | 0.15 | 0.20 | 0.25 | 0.33 | 0.50 | 0.70 |
|---|---|---|---|---|---|---|---|
| one loop | 2.667 | 2.116 | 1.537 | 1.168 | 0.799 | 0.408 | 0.199 |
| c₀ | 2.653 | 2.097 | 1.535 | 1.162 | 0.789 | 0.405 | 0.198 |

Agreement is 0.1–1.2 %. Before the finite-ε correction the same test was off by up to 10 %.

## Results

| k_split | C₂,δ at D = 1 | ΔP/P, k = 0.05 | excess over one loop | displacement C₂ |
|---|---|---|---|---|
| 0.12 | 3.304 ± 0.023 | −1.65 % | +0.64 | −1.215 |
| 0.20 | 1.935 ± 0.015 | −0.97 % | +0.40 | −0.386 |
| 0.33 | 1.057 ± 0.006 | −0.53 % | +0.26 | −0.104 |
| 0.70 | 0.314 ± 0.001 | −0.16 % | +0.12 | −0.111 |

* **Density is suppressed and displacement enhanced at every cutoff.** The README's earlier ΔP/P had used the displacement coefficient; that is corrected and published.
* **PM's C₂,δ grows nearly as D² after D ≈ 0.7,** while fixed-order density fails: PM − 3LPT at k_split = 0.2 is −1.33 at D = 1.
* **No density decoupling at ε → 0:** every shell stays 1.16–1.58× one loop at D = 1, and modes above k = 0.33 carry about a third of the response. Decoupling in §14.4 is a displacement statement.
* **Density barely sees first crossing.** PM − LPT∞ is at most 10⁻³ at D = 0.3, against up to 56 × 10⁻⁴ in displacement. At k_split ≤ 0.33 the density series has no limit at D ≥ 0.4, so the irreducibility statement rests on the displacement channel.
* **Long × short does not generalize from one cutoff.** At k_split = 0.2, D = 1 the +1.91 in displacement is about 0 in density, but random-long minus ε → 0 runs from +0.31 at k_split = 0.12 to −0.30 at 0.5, so 0.2 is near a zero crossing. The windows differ between the two.
* **a² scaling:** the a = 1 to a = 0.5 shift ratio is 4.12 ± 0.33 at D = 0.5, as two-loop predicts, then 3.43 ± 0.22 at D = 0.7, and it fails at D = 1. The raw estimator disagrees in sign at a = 0.5, D = 0.5, so that first value rests on one estimator.

## Limits

PM's particles are not stored in these runs, so its deposit bias is taken from 7LPT; where b/C₂,δ is large (k_split ≥ 0.33 at D = 0.5; 0.5 and 0.7 at D = 0.7; 0.7 at D = 1, b/C₂,δ = 0.09: marked in Table 6) values rest on that. The a > 0 runs have no exact bias calibration. b comes from 1–2 realizations. No Eulerian momentum field is stored for these runs.
