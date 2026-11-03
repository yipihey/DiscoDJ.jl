# DiscoDJ.jl

Julia tools for **DISCO-DJ**, the differentiable cosmology code of List, Hahn, Winkler & Floess
(used here through the `yipihey/DISCO-DJ` fork). The repository has two parts:

- **[DiscoDJNative](native/DiscoDJNative/README.md)**, a native Julia engine written with
  [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), so the same
  kernels run on multithreaded CPUs and on GPUs. It covers initial conditions, Lagrangian
  perturbation theory to 4th order, the particle-mesh N-body solver, the phase-space sheet,
  lightcones and power spectra. It is differentiable through ChainRulesCore/Zygote.
- **[DiscoDJLib](lib/DiscoDJLib)**, a bridge that runs the original Python/JAX code in-process
  (PythonCall), so a pipeline stays differentiable on the JAX side while Julia consumes the
  results.

## Studies

| study | question | read it |
|---|---|---|
| **LPT vs N-body before shell crossing** | How far do LPT orders reproduce N-body PDFs and correlations on scales that have almost not shell-crossed? | [report with all figures](https://yipihey.github.io/DiscoDJ.jl/) |
| **UV sensitivity of long modes across shell crossing** | Do short modes leave an irreducible effective stress on long modes, what does it look like, and how do LPT and spherical collapse compare? | [study page with all figures](https://yipihey.github.io/DiscoDJ.jl/studies/eft_shell_crossing/), [synthesis](studies/eft_shell_crossing/reports/00_synthesis.md) |

All studies include resolution studies for every method.

### LPT vs N-body before shell crossing

Identical fixed-amplitude initial conditions (L = 300 Mpc/h, top-hat filter R_f = 14 Mpc/h, Planck18). Source: [`studies/lpt_pdf/report.html`](studies/lpt_pdf/report.html). The copula study reports its validation tests, noise floors and every deviation from its protocol.

| part | what it measures | main result |
|---|---|---|
| [One-point density PDFs](studies/lpt_pdf/README.md) | Zel'dovich, 2LPT, 3LPT and 4LPT against converged N-body at z = 0 | 2LPT is too narrow in both tails: the mass-weighted 99.9 % quantile is 29 % low at 7 Mpc/h. 4LPT agrees to ≲ 1 % except in the far tails. |
| [Correlations with the PDF removed (copula)](studies/lpt_pdf/copula/report.md) | rank-Gaussianized two-point statistics on the sheet, in regions single-stream in both runs | 2LPT has a small converged error, about 5.7 × 10⁻³ in ξ at 20–30 Mpc/h; 4LPT removes 75–95 % of it. N-body's extra clustering at ~10 Mpc/h comes from its one-point PDF. |
| [Phase-space-sheet estimator](studies/lpt_pdf/sheet_pdf/README.md) | the PDFs again with exact tetrahedral sheet density, next to CIC | The PDF conclusions hold to ≤ 0.2 percentage points. In the caustic-dominated dense tail, N-body carries 4–9× more mass than 4LPT and 2LPT. |

### UV sensitivity of long modes across shell crossing

Paired 3D universes with bitwise-identical long modes and an added short-mode band, evolved with 1–7LPT and a discreteness-suppressed PM, in EdS with the Planck18 shape. Source: [`studies/eft_shell_crossing/README.md`](studies/eft_shell_crossing/README.md).

| part | what it measures | main result |
|---|---|---|
| [Main study](studies/eft_shell_crossing/README.md) | field-level long-mode response Δψ_L = −C₂k²ψ_L, the CHS (2012) cutoff running, counts in spheres, redshift space, multistream kinetic stress | The series converges onto PM before crossing (to 10⁻⁵ in C₂). Just after first crossing a coherent, pure-k², pressure-signed residual appears that higher LPT orders do not remove; PM's measured velocity-dispersion stress accounts for it to a factor ≈ 2. |
| [A: density response](studies/eft_shell_crossing/reports/A_density_response.md) | how short modes change a long density mode, ε → 0 plane waves | Modes 0.2 ≤ k < 1.2 change P(k = 0.05) by −0.97 % at D = 1, matching the one-loop kernel to ≤ 1.2 % once three effects are removed. |
| [B: two-loop stress](studies/eft_shell_crossing/reports/B_two_loop_stress.md) | the effective stress against two-loop EFT | The far-UV linear coefficient is within 1–2 % of 61/70 in Planck-shape spectra, and 20 % high in a UV-divergent one. Whether all eight counterterms are needed is undecided. |
| [C: σ(M), LPT and spherical collapse](studies/eft_shell_crossing/reports/C_sigma_M.md) | variance at fixed mass, around random and mass-drawn centres and for Lagrangian patches | At D = 1 spherical collapse holds σ(δ_R) to 5 % down to R ≈ 6 h⁻¹Mpc, against R ≈ 18 for the best LPT order; beyond D ≈ 0.5 higher LPT orders do worse. |
| [D: reach of the UV](studies/eft_shell_crossing/reports/D_uv_reach.md) | how far UV modes change the IR variance | Modes k = 0.6–1.6 change the variance by −0.58 % at R = 20 h⁻¹Mpc at D = 1, and the fraction saturates by D ≈ 0.8. |
| [E: campaign notes](studies/eft_shell_crossing/reports/E_campaign_notes.md) | settings, machine behaviour, errors made and corrected | |

## What DiscoDJNative provides

| area | contents |
|---|---|
| Cosmology | ΛCDM background, Eisenstein–Hu transfer, exact growth factors D₁–D₃ and their time derivatives |
| Initial conditions | Gaussian random fields (NGenIC-compatible), fixed and paired amplitudes |
| LPT | 1LPT–4LPT, following DISCO-DJ's `compute_core` / `compute_core_exact` with 3/2-rule de-aliasing. The fused KernelAbstractions path is differentiable and 5× faster than the reference path (4LPT at 64³: 8.6 s → 1.6 s). GPU runs through CUDA. |
| N-body | a faithful port of DISCO-DJ's `run_nbody` PM solver: BullFrog, FastPM and symplectic steppers; CIC/TSC/PCS; gradient and Laplace kernels of order 0–6; interlacing; sheet resampling. It matches the JAX code to 3 × 10⁻¹⁴ on full runs. One step at 128³ takes 0.82 s (Float64) or 0.65 s (Float32) on 4 CPU threads. |
| Phase-space sheet | tetrahedral elements with exact volumes, periodic stream counting, sheet density on a mesh, element location for mesh nodes |
| Lightcones and analysis | lightcone crossing and HEALPix shells, power spectrum, bispectrum |

## Quick start

```julia
using DiscoDJNative
c    = Cosmology("Planck18EEBAOSN")
pk   = linear_power_spectrum(c)
res, L = 128, 300.0
fphi = generate_grf(:ngenic, 3, pk, res, L, 42; dtype=Float64, dtype_c=ComplexF64)

# 2LPT initial conditions at a = 0.05, then PM N-body to z = 0
shapes = compute_core(fphi, nlpt_kernels(res, L); n_order=2)
Ψ0, Π0 = nbody_ics_lpt(c, shapes, 0.05; n_order=2)
X, P, a = run_nbody(c, Ψ0, Π0; boxsize=L, a_ini=0.05, a_end=1.0, n_steps=64,
                    res_pm=2res, stepper=:bullfrog, time_var=:D)
```

See [the DiscoDJNative README](native/DiscoDJNative/README.md) for GPU use, half-precision
storage, performance tables and the N-body options. Run the test suite with
`julia --project=native/DiscoDJNative native/DiscoDJNative/test/runtests.jl`.

The JAX bridge:

```julia
ENV["JULIA_PYTHONCALL_EXE"] = expanduser("~/Projects/disco-dj-fem/.venv/bin/python")
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"          # before using DiscoDJLib
using DiscoDJLib
b  = build(DiscoSpec(res = 128, boxsize = 100.0, n_order = 3, seed = 42))   # live JAX object
ic = lpt_ics(b, 0.02)                            # ψ, pos, vel as Julia arrays
```

## Repository layout

| path | contents |
|---|---|
| `native/DiscoDJNative/` | the native Julia engine (`src/cosmology`, `ics`, `lpt`, `nbody`, `field`, `lightcone`, `analysis`) and its tests, including the parity tests against JAX DISCO-DJ |
| `lib/DiscoDJLib/` | the PythonCall bridge to JAX DISCO-DJ |
| `studies/lpt_pdf/` | the one-point PDF study, the HTML report and the site build script |
| `studies/lpt_pdf/copula/` | the copula study: pipeline, configuration, results and figures |
| `studies/lpt_pdf/sheet_pdf/` | the phase-space-sheet estimator study (exact R3D remeshing, tetrahedron mass weighting) |
| `studies/eft_shell_crossing/` | the EFT / shell-crossing study: paired-IC experiments, analysis, CHS Fig. 3 reconstruction, two-loop stress, σ(M), figures, `reports/` |

## Roadmap

- Multi-GPU and Metal support for the N-body solver (Metal needs a host or third-party FFT).
- The remaining DISCO-DJ force options (`nufftpm`, `treepm`) and an adjoint for `run_nbody`.
- DiscoDJ as an initial-conditions source for the other codes in the EnzoNG.jl federation,
  with a three-way IC cross-check against MUSIC and NGenIC.
