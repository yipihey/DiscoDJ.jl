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
- Nbody:     PM N-body (BullFrog/FastPM) — deferred
"""
module DiscoDJNative

using FFTW

# Enable FFTW multi-threading as soon as the module loads.
function __init__()
    FFTW.set_num_threads(Threads.nthreads())
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

end # module
