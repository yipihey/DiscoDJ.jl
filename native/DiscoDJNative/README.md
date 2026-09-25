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

`store=:f16` builds each ψ **directly as a `HalfField`** — every component is
packed to f16 (minus its mean) as soon as it is computed, reusing one f32 scratch,
so the full f32 `(res,res,res,3)` ψ is *never* materialised. That lowers the
displacement *compute* peak (not just the stored size), and the evaluators
(`evaluate_lpt_*`) accept packed results transparently (`pack_half`/`expand_half`
expose the conversion).

This is what lets **2LPT@1024³ fit on the 48 GB A6000** (≈26 GiB working set; the
f32 path OOMs there). The displacement footprint halves (6 vs 12 bytes/cell) at
~3e-4 round-off.

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

**Grid memory.** Two reductions shrink the per-grid GPU footprint (896³: **16.1 →
9.4 GiB**, −6.7 GiB):
- The `kx,ky,kz` components are separable, so the grid keeps them as compact
  reshaped 1-D vectors that broadcast on use — no three dense `(res,res,res÷2+1)`
  arrays (−4.3 GiB at 896³), and the elementwise kernels read O(res) k-values
  instead of O(res³). (k² stays dense — it isn't separable and `inv_laplace` reads
  it linearly.)
- `to_gpu(grid)` frees the r2c plan's preservation buffer — one complex
  half-spectrum (2.7 GiB at 896³) that CUDA.jl allocates but the forward-only
  `mul!` path never uses.

What remains of the per-grid cuFFT cost is the **internal workspace** (not exposed
by CUDA.jl), which scales with the resolution's prime factorisation: small for
smooth sizes (2ᵃ·3ᵇ·5ᶜ), large for big factors (e.g. 896 = 2⁷·7, 832 = 2⁶·13).
**Prefer smooth resolutions** — they cut both the workspace and the FFT time.

**Crossover:** with these reductions 2LPT fits to ~896³ and 3LPT to ~768³ on the
A6000 (vs ~768³/~640³ before); the exact edge near 44 GiB is set by the cuFFT
workspace, so a non-smooth size can OOM below a larger smooth one. Larger boxes
use the CPU / 2 TB RAM. With `store=:f16` (incremental packing) **2LPT reaches
1024³** on the A6000.

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

## N-body (`src/nbody/`)

A faithful port of DISCO-DJ's `run_nbody(method="pm")` (JAX, `yipihey/disco-dj`), written with
KernelAbstractions so the scatter/gather kernels run on CPU and CUDA (FFTs through the pipeline's
`_rfftn`/`_irfftn`, overridden by the CUDA extension; Metal.jl has no FFT, so a host FFT would be
needed there).

```julia
c = Cosmology("Planck18EEBAOSN")
shapes = compute_core(fphi, nlpt_kernels(res, L); n_order=2)        # DISCO-DJ nLPT
Ψ0, Π0 = nbody_ics_lpt(c, shapes, 0.05; n_order=2)                  # Ψ and Π = dΨ/dD at a_ini
X, P, a = run_nbody(c, Ψ0, Π0; boxsize=L, a_ini=0.05, a_end=1.0, n_steps=10,
                    res_pm=2res, stepper=:bullfrog, time_var=:D)
```

Ported: steppers `:bullfrog`, `:fastpm`, `:symplectic` (drift–kick–drift, DISCO-DJ coefficients);
time variables `:a`, `:log_a`, `:D`, `:superconft` or an explicit a-grid; PM force options
`worder` 2/3/4, `deconvolve`, `antialias` −1/0/1/2/3, gradient and inverse-Laplacian kernels of
order 0/2/4/6, `n_resample` sheet resampling (`:fourier`, `:linear`); LPT and BullFrog initial
conditions; `collect_all`, `return_all_a`, `return_displacement`, `step_callback`.
Not ported: `nufftpm`, `treepm`, the 1-D exact force, Diffrax and the custom-VJP adjoint.

**Parity:** `test/nbody_parity_tests.jl` compares against reference data exported from the JAX
code (`test/reference/export_jax_nbody_reference.py`, float64, 12 configurations): timetables
≲ 4e-15, LPT initial conditions ≲ 5e-16, PM accelerations ≲ 1.2e-14, stepper coefficients
≲ 6e-15, and full runs (Ψ, P) ≲ 3e-14.

**Upstream bug (not ported):** DISCO-DJ's `interpolate_field(which="linear")` evaluates the
resampled sheet at grid index `(q/L_unit + dshift·L/res)/L·res`, mixing box units; it is only
correct for `boxsize = 1`.  The port uses the intended index `i + dshift`; the parity test for
`resampling=:linear` compares against the JAX code with that line corrected.

**Cold-lattice instability (inherited, by design of the options):** an `ik` gradient
(`grad_kernel_order=0`) with `deconvolve=true` on a PM mesh finer than the particle lattice
amplifies lattice modes from round-off; use the default 4th-order gradient without
deconvolution, or the BullFrog-IC settings (interlacing + sheet resampling).
