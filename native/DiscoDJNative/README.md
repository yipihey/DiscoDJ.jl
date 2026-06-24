# DiscoDJNative

Native-Julia port of DISCO-DJ's LPT/IC/analysis pipeline, written on
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) so the
same kernels run on the CPU (multithreaded) and the GPU (CUDA/cuFFT).

The nLPT pipeline is **FFT-bound** (2LPT ≈ 13 transforms of size `res³`, 3LPT
many more), so performance is dominated by the FFT backend and memory bandwidth.

## CPU

```julia
using DiscoDJNative
c    = Cosmology("Planck18EEBAOSN")
pk   = linear_power_spectrum(c)
res  = 256; L = 1000.0; T = Float32          # Float32 ≈ 1.5× faster, ½ the memory
grid = get_fourier_grid(res, L; T=T)
fphi = generate_grf(:ngenic, 3, pk, res, L, 42; dtype=T, dtype_c=Complex{T})
lpt  = compute_lpt(fphi, grid; n_order=2, backend=:threads)   # :threads or :ka
```

Tuning:
- **Threads.** Launch with `JULIA_NUM_THREADS` set; FFTW inherits the count. The
  pipeline is memory-bandwidth-bound, so it scales sub-linearly and saturates at
  about one socket — on the dual-EPYC-7763 box the sweet spot is **~32 threads**
  (2LPT@256³: 2.0 s @8t → 1.6 s @32t, no gain beyond). Pin with `numactl` to keep
  memory NUMA-local.
- **FFT planner.** Default is `FFTW.ESTIMATE` (race-free). `MEASURE` is ~25-30 %
  faster but on this FFTW.jl/Julia-1.12 stack its extra threaded planning
  intermittently segfaults in FFTW's `spawnloop`; it is opt-in via
  `DISCODJ_FFTW_PLANNER=measure` (plans cached as on-disk wisdom). Most of its
  benefit is also reachable safely just by using more FFT threads.
- `backend=:threads` vs `:ka` are close on CPU; `:threads` is the safe default.

## GPU (CUDA)

```julia
using DiscoDJNative, CUDA
grid  = get_fourier_grid(res, L; T=Float32)
fphi  = generate_grf(:ngenic, 3, pk, res, L, 42; dtype=Float32, dtype_c=ComplexF32)
gridg = to_gpu(grid)                         # k-grids + cuFFT plans → device
fphig = to_gpu(fphi)                         # canonicalised + uploaded
lpt   = compute_lpt(fphig, gridg; n_order=2, backend=:ka)   # runs on the A6000
psi1  = to_host(lpt.psi1)                     # (res,res,res,3) host array, x,y,z order
```

`using CUDA` loads the `DiscoDJNativeCUDAExt` extension. `compute_lpt` is
unchanged — the KA kernels infer their backend from the arrays, the temporaries
are allocated `similar` to the (device) input, and the FFTs use the grid's cuFFT
plans.

Two device-specific details, both handled by `to_gpu`:
- **FFT layout.** cuFFT requires the rfft reduced axis first (`region [1,2,3]`),
  while the CPU keeps it last (`[3,1,2]`); `to_gpu` permutes `(3,1,2)` in and
  `to_host` permutes back. The run between is elementwise/separable, so it is
  numerically identical.
- **Hermitian canonicalisation.** `generate_grf` (NGenIC-style) leaves the
  DC/Nyquist planes non-canonical; FFTW's C2R tolerates this, cuFFT's does not.
  `to_gpu` round-trips the spectrum (`canonicalize_hermitian`, idempotent under
  irfft → same physics) so both libraries agree. To compare CPU and GPU exactly,
  feed the CPU run `canonicalize_hermitian(fphi, grid)` too.

  A residual ~1e-3 (res 128) shrinking with resolution remains from the
  Nyquist-derivative convention (`i·k·φ` is imaginary at self-conjugate modes,
  which FFTW and cuFFT resolve differently) — physically negligible: Nyquist
  modes carry vanishing power in a CDM spectrum.

On the shared A6000 set a memory ceiling, e.g. `JULIA_CUDA_HARD_MEMORY_LIMIT=38GiB`.

## Half-precision field storage (`store=:f16`)

The displacement components are near-zero-mean with a modest spread, so storing a
per-component Float32 mean plus a Float16 residual reproduces them to ~3e-4 — well
below LPT's own accuracy — at **half** the footprint (6 vs 12 bytes/cell).

```julia
lpt = compute_lpt(fphig, gridg; n_order=2, backend=:ka, store=:f16)
lpt.psi1 isa HalfField          # per-component f32 mean + f16 residual
psi = evaluate_lpt_psi_at_a(lpt, c, 0.02)   # transparently expands to f32
```

`store=:f16` packs each ψ as it is computed and frees the f32 source, so the held
displacement memory halves. `pack_half`/`expand_half` expose the conversion; the
evaluators (`evaluate_lpt_*`) accept packed results transparently.

This directly doubles how large a field you can **hold/output** on the GPU, and
near-doubles the box for transient-light orders (1LPT/Zel'dovich), and stacks on
top of the memory-lean source construction below.

## Performance (RTX A6000 vs dual EPYC 7763, 2LPT, Float32)

| res  | CPU (32t) | GPU      | speedup | GPU throughput |
|-----:|----------:|---------:|--------:|---------------:|
| 128³ |   218 ms  |  3.1 ms  |   71×   |  685 Mcell/s   |
| 256³ |  1.63 s   | 23.5 ms  |   69×   |  713 Mcell/s   |
| 512³ | 11.4 s    |  181 ms  |   63×   |  742 Mcell/s   |

The GPU sustains ~700-740 Mcell/s (memory-bandwidth + cuFFT bound) vs the CPU's
~10-12 Mcell/s.

**Memory-lean source construction.** The 2LPT/3LPT sources are built by streaming
the second derivatives through reused buffers (`mul!`, no per-FFT allocation):
2LPT via the trace identity `S₂ = ½[(tr H)² − tr(H²)]` accumulates from one
derivative at a time (**3** real buffers, was 7), and 3LPT holds the six `d1`
needed for `det(H₁)` but streams the six `d2` cross-terms (**8** real buffers, was
15). Measured device working sets: 2LPT@832³ ≈ 27 GiB, 3LPT@768³ ≈ 34 GiB — both
fit the 48 GB A6000, where the un-streamed 3LPT@768³ did not.

**Crossover:** 2LPT fits to ~896³ and 3LPT to ~768³ on the A6000 (the exact edge
near 44 GiB is sensitive to cuFFT plan workspace); larger boxes use the CPU / 2 TB
RAM. `store=:f16` stacks on top (e.g. 3LPT@704³: 34 → 26 GiB).

## Benchmarks & tests

```bash
# CPU pipeline + FFT/threads probes (package env)
JULIA_NUM_THREADS=32 julia --project=. test/bench/bench_pipeline.jl 128 256
JULIA_NUM_THREADS=32 julia --project=. test/bench/bench_threads.jl 256

# CPU↔GPU crossover (needs an env with DiscoDJNative + CUDA)
JULIA_NUM_THREADS=32 JULIA_CUDA_HARD_MEMORY_LIMIT=38GiB \
  julia --project=<gpu-env> test/bench/bench_cpu_gpu.jl 128 256 512

# Tests (47 CPU; +12 GPU when CUDA is functional)
julia --project=. test/runtests.jl
```
