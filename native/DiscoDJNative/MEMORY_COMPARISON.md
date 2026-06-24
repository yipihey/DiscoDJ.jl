# GPU memory: JAX DISCO-DJ vs. the `DiscoDJNative` rewrite

Peak GPU memory for the **nLPT displacement-field computation**, comparing the
original JAX **DISCO-DJ** (the `yipihey/DISCO-DJ` fork with the lightcone work)
against this pure-Julia **`DiscoDJNative`** rewrite. Same physics, same single
precision, same box, same hardware.

## TL;DR

- At a matched resolution and LPT order, the rewrite uses **1.3–2.7× less** GPU
  memory in Float32, and **1.2–3.9× less** with `store=:f16` — and the gap
  **widens with resolution and LPT order**.
- That moves the largest box that fits a **48 GB RTX A6000** by a wide margin:

  | LPT order | JAX max box | Ours (f32) | Ours (f16) | gain in particle count |
  |:--|:--:|:--:|:--:|:--:|
  | 1LPT (Zel'dovich) | 768³ | **1024³** | **1024³** | ~2.4× |
  | 2LPT | 512³ | 768³ | **1024³** | **~8×** |
  | 3LPT | 256³ | **768³** | **768³** | **~27×** |

  e.g. JAX DISCO-DJ tops out at **256³ for 3LPT** on this GPU; the rewrite runs
  **768³** — 27× more particles.

## Full results — peak GPU memory (GiB, total device footprint)

`OOM` = exceeded the 48 GB A6000 (≈45 GiB usable). Ratios are JAX ÷ ours.

| order | res | JAX | Ours f32 | Ours f16 | JAX/f32 | JAX/f16 |
|:--:|--:|--:|--:|--:|:--:|:--:|
| 1LPT | 128³ | 0.5 | 0.3 | 0.4 | 1.3× | 1.2× |
| 1LPT | 256³ | 1.5 | 0.8 | 0.7 | 2.0× | 2.1× |
| 1LPT | 512³ | 8.8 | 4.1 | 3.3 | 2.2× | 2.6× |
| 1LPT | 768³ | 24.0 | 13.0 | 10.5 | 1.8× | 2.3× |
| 1LPT | 1024³ | **OOM** | 30.3 | 24.3 | — | — |
| 2LPT | 128³ | 0.6 | 0.4 | 0.4 | 1.6× | 1.6× |
| 2LPT | 256³ | 2.9 | 1.2 | 0.9 | 2.5× | 3.2× |
| 2LPT | 512³ | 18.8 | 7.1 | 4.8 | **2.7×** | **3.9×** |
| 2LPT | 768³ | **OOM** | 23.1 | 15.5 | — | — |
| 2LPT | 1024³ | **OOM** | OOM | 36.4 | — | — |
| 3LPT | 128³ | 0.7 | 0.5 | 0.5 | 1.6× | 1.6× |
| 3LPT | 256³ | 3.7 | 1.6 | 1.4 | 2.4× | 2.7× |
| 3LPT | 512³ | **OOM** | 13.2 | 8.6 | — | — |
| 3LPT | 768³ | **OOM** | 41.8 | 28.2 | — | — |
| 3LPT | 1024³ | OOM | OOM | OOM | — | — |

**Per-cell footprint** (large-res, where the fixed context overhead is amortised):
JAX ≈ **150 bytes/cell** for 2LPT, the rewrite ≈ **55 B/cell** (f32) / **37 B/cell**
(f16) — a ~3–4× reduction in the marginal memory cost.

## Why the rewrite is lighter

The JAX pipeline is functional and `jit`-traced: XLA does its own buffer
planning, but DISCO-DJ's nLPT recursion materialises the intermediate
second-derivative and source fields (and the dense Fourier k-grids), and that set
grows with order — which is why JAX's footprint climbs steeply (2LPT ≈ 150 B/cell,
3LPT OOMs already at 512³). The rewrite was built to minimise exactly that working
set:

- **Streamed source construction.** 2LPT uses the trace identity
  `S₂ = ½[(tr H)² − tr(H²)]` to accumulate from one second-derivative at a time
  (3 real buffers instead of 7); 3LPT holds the six `d1` that `det(H₁)` needs but
  streams the six `d2` cross-terms (8 instead of 15). All transforms run through
  reused buffers via `mul!` (no per-FFT allocation).
- **Compact k-vectors.** `kx,ky,kz` are stored as reshaped 1-D vectors and
  broadcast on use, so three dense `(res,res,res÷2+1)` k-arrays never exist
  (−4.3 GiB at 896³).
- **No wasted cuFFT buffer.** The r2c plan's unused preservation buffer is freed.
- **Incremental f16 storage.** `store=:f16` packs each displacement component to
  Float16 (minus a Float32 mean) as it is computed, so the full Float32 ψ is never
  materialised — at ~3e-4 round-off. This is what lets 2LPT reach 1024³.

## Setup & methodology

- **Hardware:** 1× NVIDIA RTX A6000 (48 GB; ≈45 GiB usable), shared host.
- **JAX side:** `yipihey/DISCO-DJ` @ `20e774c`, JAX 0.10.2 (`jax[cuda12]`),
  `precision="single"`, `device="gpu"`. Pipeline:
  `DiscoDJ(...).with_timetables().with_linear_ps().with_ics().with_lpt(n).evaluate_lpt_psi_at_a`.
- **Julia side:** `DiscoDJNative` @ branch `claude/discodj-feature-audit-06ttwx`,
  `compute_lpt(...; n_order, backend=:ka, store)`, `Float32`.
- **Metric:** total device memory (driver-level, `nvidia-smi memory.used`) at the
  computation's retained peak — the all-inclusive "GPU you need" (CUDA/XLA context
  + framework libraries + FFT workspace + field buffers). For JAX this is its
  CUDA-context baseline plus XLA's `peak_bytes_in_use` (which includes cuFFT
  scratch); for Julia it is the pool high-water of a single warm `compute_lpt`
  after clearing allocator churn. Both run `XLA_PYTHON_CLIENT_PREALLOCATE=false` /
  no Julia hard limit so the number reflects real usage, not a preallocation.
- Both carry their inputs/grid; box = 1000 Mpc/h, seed 42, Eisenstein–Hu transfer,
  Planck18 cosmology. Run-to-run measurement noise is a few hundred MiB (context
  jitter), negligible at the resolutions where memory matters.

### Caveats / fairness

- Compared workload is the **nLPT field computation** (the common, memory-dominant
  core), not the PM N-body or the lightcone post-processing.
- JAX numbers include DISCO-DJ's resident pipeline state (timetables, linear P(k),
  white noise) — all small 1-D/1× arrays; the 3-D fields dominate, so this does
  not materially affect the comparison.
- Neither side requests gradients (no adjoint tape), so both are forward-only.
- `store=:f16` is a storage option unique to the rewrite; the f32 columns are the
  strict apples-to-apples comparison.

*Reproduce:* harnesses are in `test/bench/` — `bench_mem.jl <res> <order> <f32|f16>`
(Julia, needs an env with `DiscoDJNative`+`CUDA`) and `bench_mem_jax.py <res> <order>`
(run inside the DISCO-DJ checkout's JAX venv, with `XLA_PYTHON_CLIENT_PREALLOCATE=false`).
