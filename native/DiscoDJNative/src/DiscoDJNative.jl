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

# ── Cosmology ────────────────────────────────────────────────────────────────
include("cosmology/Cosmology.jl")
include("cosmology/transfer.jl")
include("cosmology/growth.jl")

# ── Initial conditions ───────────────────────────────────────────────────────
include("ics/ngenic.jl")
include("ics/grf.jl")

# ── LPT ─────────────────────────────────────────────────────────────────────
include("lpt/grids.jl")
include("lpt/kernels_ka.jl")
include("lpt/kernels_threads.jl")
include("lpt/nlpt.jl")
include("lpt/evaluate.jl")

# ── Lightcone ────────────────────────────────────────────────────────────────
include("lightcone/shells.jl")
include("lightcone/replicas.jl")
include("lightcone/crossing.jl")
include("lightcone/sky.jl")
include("lightcone/healpix.jl")
include("lightcone/io.jl")
include("lightcone/lightcone.jl")
include("lightcone/refresh.jl")

# ── Analysis ─────────────────────────────────────────────────────────────────
include("analysis/power_spectrum.jl")
include("analysis/bispectrum.jl")

end # module
