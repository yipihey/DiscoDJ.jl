"""
    EFTShell

Paired-initial-condition experiments on the UV→IR response of collisionless dark matter
(the "irreducible effective stress" question of Carrasco, Hertzberg & Senatore 2012), built on
DiscoDJNative: general-order nLPT (`compute_core`) and the DISCO-DJ PM N-body port
(`run_nbody`), both on CUDA.

Files:
  ics.jl         paired white noise with an exact sharp-k split (shared long modes)
  measure.jl     GPU measurements: low-k Fourier modes of displacement / velocity / density,
                 inverted Lagrangian tetrahedra (shell crossing), conservation checks
  runners.jl     LPT (orders 1..n_max at a list of epochs) and PM runs of one realization
  fixedmass.jl   fixed-mass sheet statistics: F(R,M), its R/D derivatives, real-space mass budget
  estimators.jl  response R(k), counterterm fits C2/C4, stochastic power (pure CPU, tested)
"""
module EFTShell

using DiscoDJNative, FFTW, Random, LinearAlgebra, Statistics, Printf
using KernelAbstractions
const DJ = DiscoDJNative

include("ics.jl")
include("measure.jl")
include("fixedmass.jl")
include("runners.jl")
include("estimators.jl")
include("analysis.jl")

end
