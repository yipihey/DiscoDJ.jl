"""
    DiscoDJNative

Native Julia port of DISCO-DJ (yipihey/DISCO-DJ fork) — feature-complete
rewrite using KernelAbstractions.jl for GPU/CPU portability.

Modules (load order matters for dependencies):
- Cosmology: background cosmology, growth factors, transfer functions
- ICs:       NGenIC-compatible white noise + Gaussian random field generation
- LPT:       nLPT displacement fields with KA + Threads dual backends
- Lightcone: past-lightcone catalogues, refresh, sky maps
- Analysis:  power spectrum, cross-spectrum, bispectrum
- Nbody:     PM N-body — faithful port of DISCO-DJ run_nbody (BullFrog/FastPM/symplectic DKD,
             PM force with CIC/TSC/PCS, deconvolution, interlacing, sheet resampling), KA kernels
"""
module DiscoDJNative

using FFTW

# ── FFTW threading: thread cap + a migration-safe spawn callback ──────────────
# FFTW.jl hooks FFTW's threads to Julia with a `spawnloop` callback that `@spawn`s one task per chunk
# and waits in `@sync`.  Measured on this stack (Julia 1.12.6, FFTW.jl 1.10, FFTW_jll 3.3.12, 2× EPYC
# 7763, julia -t 96, 384³ Float32 r2c+c2r):
#  * > 64 FFTW threads is broken: the stock callback segfaults in `spawn_apply` (deterministic at 80
#    and 96) or runs ~100× slower (128); a pinned callback also runs 60–170× slower.  ≤ 64 threads:
#    0.36–0.45 s per pair for every callback.  ⇒ FFTW threads are capped at `FFTW_MAX_THREADS = 64`.
#  * FFTW re-enters the callback from its own worker tasks (~10⁵ nested calls per 512³ transform) and,
#    with the stock callback, a worker waiting in `@sync` with FFTW's C frames on its stack resumes on
#    another OS thread (~10⁴ migrations per transform); production CPU runs segfaulted intermittently
#    in `spawn_apply` even at 16 threads.  Here every waiting task is pinned (sticky) while it waits:
#    0 migrations measured, same speed, bitwise identical transforms.
function _fftw_safe_spawnloop(f::Ptr{Cvoid}, fdata::Ptr{Cvoid}, elsize::Csize_t, num::Cint, ::Ptr{Cvoid})
    t = current_task(); sticky = t.sticky; t.sticky = true
    try
        @sync for i in 0:num-1
            Threads.@spawn ccall(f, Ptr{Cvoid}, (Ptr{Cvoid},), fdata + elsize * i)
        end
    finally
        t.sticky = sticky
    end
    return nothing
end

const FFTW_MAX_THREADS = 64
"""Set FFTW's thread count, capped at `FFTW_MAX_THREADS` (see the note above `_fftw_safe_spawnloop`)."""
fftw_set_threads(n::Integer=Threads.nthreads()) = FFTW.set_num_threads(clamp(n, 1, FFTW_MAX_THREADS))

const _FFTW_SPAWN_CB = Ref{Ptr{Cvoid}}(C_NULL)

function _install_fftw_spawnloop()
    (FFTW.fftw_provider == "fftw" && Threads.nthreads() > 1) || return false
    _FFTW_SPAWN_CB[] = @cfunction(_fftw_safe_spawnloop, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint, Ptr{Cvoid}))
    ccall((:fftw_threads_set_callback, FFTW.libfftw3_no_init), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), _FFTW_SPAWN_CB[], C_NULL)
    ccall((:fftwf_threads_set_callback, FFTW.libfftw3f_no_init), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), _FFTW_SPAWN_CB[], C_NULL)
    return true
end

# Enable FFTW multi-threading as soon as the module loads.  `set_num_threads` triggers FFTW.jl's (lazy)
# thread initialisation, which installs its own callback; ours replaces it afterwards.
function __init__()
    fftw_set_threads()
    _install_fftw_spawnloop()
end

# ── GPU interface ─────────────────────────────────────────────────────────────
# Implemented by the CUDA package extension (ext/DiscoDJNativeCUDAExt.jl); call
# `using CUDA` to enable.  `to_gpu` moves a FourierGrid (k-grids + cuFFT plans)
# or a Fourier-space field onto the device, applying the (3,1,2) layout swap that
# puts the rfft half-axis on dim 1 — cuFFT requires the reduced axis first, while
# CPU code keeps it on dim 3.  `to_host` brings displacement fields back and
# undoes that swap.  Both error helpfully until `using CUDA`.
to_gpu(::Any) = error("to_gpu requires the CUDA extension — run `using CUDA` first.")
to_host(x) = x                      # no-op on host; CUDA ext adds CuArray methods
export to_gpu, to_host

# Release a device buffer eagerly (so packed-to-f16 fields free their f32 source
# and lower the GPU peak).  No-op on host; the CUDA ext frees CuArrays.
_free!(x) = nothing

# ── Cosmology ────────────────────────────────────────────────────────────────
include("cosmology/Cosmology.jl")
include("cosmology/transfer.jl")
include("cosmology/growth.jl")
include("nbody/timetables.jl")     # jnp.interp-exact timetable lookups for the N-body steppers

# ── Initial conditions ───────────────────────────────────────────────────────
include("ics/ngenic.jl")
include("ics/ngenic_gsl.jl")     # bit-exact N-GenIC (GSL ranlxd1) — reproduce a seed
include("ics/grf.jl")

# ── LPT ─────────────────────────────────────────────────────────────────────
include("lpt/grids.jl")
include("lpt/halffield.jl")
include("lpt/kernels_ka.jl")
include("lpt/kernels_threads.jl")
include("lpt/nlpt.jl")
include("lpt/evaluate.jl")
include("lpt/nlpt_ad.jl")        # differentiable (AD-traceable) nLPT path
include("lpt/nlpt_core.jl")      # faithful general-order nLPT port (parity reference)
include("lpt/nlpt_fast.jl")      # fast engine: shared ext derivative fields + fused KA source kernels

# ── N-body (port of DISCO-DJ run_nbody: PM force + DKD BullFrog/FastPM/symplectic steppers) ──
include("nbody/pm.jl")
include("nbody/run_nbody.jl")

# ── Field deposit (differentiable density estimators) ─────────────────────────
include("field/sheet_deposit.jl")  # CIC + tetrahedral CDM-sheet deposit (+ rrule)
include("field/sheet_density.jl")  # grid-free AHK sheet density — per-tet core (P1)
include("field/sheet_density_masked.jl")  # sheet-on-mask: AHK nodal density restricted to footprint trace-back
include("field/gs_poisson.jl")     # differentiable red-black Gauss-Seidel Poisson smoother
include("field/multigrid.jl")      # FFT-free geometric multigrid Poisson (+ AMR mask) — resolution unlock

# ── Lightcone ────────────────────────────────────────────────────────────────
include("lightcone/shells.jl")
include("lightcone/replicas.jl")
include("lightcone/crossing.jl")
include("lightcone/crossing_ad.jl")  # differentiable lightcone crossing (IFT) for inference
include("lightcone/sky.jl")
include("lightcone/healpix.jl")
include("lightcone/io.jl")
include("lightcone/lightcone.jl")
include("lightcone/refresh.jl")

# ── Analysis ─────────────────────────────────────────────────────────────────
include("analysis/power_spectrum.jl")
include("analysis/bispectrum.jl")

# ── Sheet query: density + deformation eigenvalues at arbitrary points ─────────
include("field/sheet_query.jl")  # our-ICs → evolve sheet → fast point query (ρ, ∂x/∂q eigenvalues)
include("field/sheet_periodic.jl")  # periodic-box sheet: element volumes/flips, exact stream counts, mesh density
include("field/sheet_moments.jl")   # periodic sheet: stream-resolved mean velocity + dispersion tensor at mesh nodes

end # module
