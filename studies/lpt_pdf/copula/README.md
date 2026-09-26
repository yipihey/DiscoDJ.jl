# Copula study: LPT vs N-body two-point statistics with the one-point PDF removed

Results and their interpretation are in [report.md](report.md). This file covers how to
reproduce them.

Every statistic in this study is **conditional on the single-stream mask** (both runs) and
is not an unconditional two-point statistic.

## Pipeline

All displacement fields and all phase-space-sheet products come from **DiscoDJNative**
(`native/DiscoDJNative`, Julia, KernelAbstractions):

- nLPT via `compute_core_exact` (orders ≤ 3, exact growth) and `compute_core` (4th order);
- N-body via `run_nbody` (the port of DISCO-DJ's BullFrog / FastPM / symplectic steppers;
  parity with JAX DISCO-DJ to 3e-14);
- sheet products via the periodic AHK kernels in `src/field/sheet_periodic.jl`.

Python handles the IC potential, rank-Gaussianization, masked two-point functions and the
bookkeeping.

| stage | command | writes |
|---|---|---|
| snapshots | `./make_snapshots.sh` | `_scratch/copula/psi_<model>_N<N>_z<z>.npy` |
| step 1 (validation) | `./run_step1.sh` (resumable) | `results/step1/*.json`, `results/step1_tables.md`, `figures/step1_*` |
| steps 2–5 | `./run_steps.sh` (resumable) | `results/steps/steps_N<N>_l<level>_z<z>.npz` |
| collect 2–5 | `python3 analysis.py collect` | `results/steps/summary.json`, `results/steps_tables.md`, `figures/steps_*.{png,pdf}` |
| everything | `./run_all.sh` | step 1, then a gate that requires every pass/fail test to pass, then steps 2–5 |

Individual units, one command per test, resolution, z and refinement level:

```
python3 validate.py a N LEVEL              # 1a  z_init: LPT vs N-body element by element
python3 validate.py b N LEVEL              # 1b  invariance under monotone transforms (z = 0)
python3 validate.py c N LEVEL Z [SEEDS]    # 1c  tie-seed and refinement noise floors
./run_test_d_band.sh && python3 validate.py d   # 1d  Zel'dovich vs Doroshkevich | det J > 0
python3 analysis.py run N LEVEL Z          # steps 2-5 for one (N, level, z), both LPT orders
```

Environment:
- Julia 1.12 with `--project=.`, which resolves `Project.toml` / `Manifest.toml` here and
  uses DiscoDJNative as a dev dependency;
- Python 3 with numpy, scipy, numba, h5py and matplotlib;
- `JULIA`, `JULIA_DEPOT_PATH` and `COPULA_SCRATCH` / `COPULA_OUT` / `COPULA_FIG` override the
  default locations.

## Configuration and seeds

Everything is in [config.json](config.json), and every result file embeds a copy of it.

| item | value |
|---|---|
| box, cosmology | L = 300 Mpc/h, Planck18 (EE+BAO+SN) |
| ICs | fixed amplitudes, real-space top-hat filter R_F = 14 Mpc/h, phase seed 1 |
| particle loadings | N = 64, 128, 256 (particle-number convergence) |
| sheet refinement | levels 0 and 1 (Fourier-interpolated displacement). Level 1 at N = 256 does not fit this machine (15 GB RAM), so refinement convergence is measured at N = 64 and 128. |
| redshifts | z = 24 (a_initial), 1, 0 |
| smoothing R | 0 (element density), 1.5, 3, 7, 14 Mpc/h, i.e. below, near and above the interparticle spacing (4.7 / 2.3 / 1.2 Mpc/h) |
| LPT orders compared | 2LPT, 4LPT |
| N-body | BullFrog, 64 steps (D-uniform, snapshot nodes), PM mesh 2N, 4th-order gradient, CIC, no deconvolution, 2LPT ICs at a = 0.04 |
| Lagrangian values R > 0 | top-hat smoothed sheet density on a 512³ Eulerian mesh, sampled at element centroids |
| Eulerian analysis mesh | 256³ (512³ ran out of memory at N = 256); Eulerian R = 1.5 is limited by the mesh |
| tie-break jitter seeds | 11 (analysis), 11/12/13 (step 1c floor) |
| test 1d | 32 band realisations (seeds 101–132) at N = 128, level 1; 2e7 Monte-Carlo draws (seed 7) |

## Data products

- `_scratch/copula/` (not in git): snapshots and per-element / per-mesh HDF5 products.
  These are large, so products are deleted after each (N, z, level) group.
- `results/`, `figures/`: JSON / NPZ results, generated tables, and PNG + PDF figures.

## Tests

- `python3 test_sheet.py`: tet volumes, Gaussianization and masked ξ against brute force
  (1e-17), and the Julia vs numba sheet cross-check.
- `native/DiscoDJNative/test/runtests.jl`: the DiscoDJNative suite (sheet, nLPT fast vs
  lean, N-body parity against JAX).
