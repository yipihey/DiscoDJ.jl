#!/bin/bash
# Reproduce the EFT / shell-crossing study end to end.
#
#   EFTSC_SCRATCH=/path/to/scratch ./run_all.sh            (≈ 7 GPU-hours on one RTX A6000)
#
# Raw per-job files (compact low-k modes, ≈ 2–25 MB each) go to $EFTSC_SCRATCH/<config>/; the reduced
# statistics committed in results/ and the figures in figures/ are regenerated from them.
set -euo pipefail
cd "$(dirname "$0")"
export EFTSC_SCRATCH=${EFTSC_SCRATCH:-/zpool/nvme/data/tabel_scratch/eft_shell_crossing}
export JULIA_CUDA_HARD_MEMORY_LIMIT=${JULIA_CUDA_HARD_MEMORY_LIMIT:-44GiB}
PY=${PY:-python3}                                   # needs numpy, h5py, matplotlib

julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -t 8 test/runtests.jl            # IC pairing, null pair, LPT parity in ±L, tetrahedra, estimators

# 1. convergence tests (single realization L1/S1001, +L)
MODEL=lpt scripts/run_queue.sh config/conv_lpt256.toml config/conv_lpt384.toml config/conv06_lpt256.toml
MODEL=pm  scripts/run_queue.sh config/conv_pm256m2.toml config/conv_pm384m2.toml config/conv_pm384m2s400.toml \
          config/conv_pm512m15.toml config/conv_pm512m2lean.toml config/conv_pm384m2rs.toml \
          config/conv_pm256m2bf.toml config/conv_pm384m2bf.toml config/conv06_pm256m2.toml config/conv06_pm384m2.toml
for c in conv_lpt256 conv_lpt384 conv06_lpt256; do julia --project=. scripts/analyze.jl $c; done
julia --project=. scripts/residual_response.jl conv_lpt256 conv_pm256m2 conv_pm384m2 conv_pm384m2s400 conv_pm512m15 \
      conv_pm512m2lean conv_pm384m2rs conv_pm256m2bf conv_pm384m2bf
julia --project=. scripts/residual_response.jl conv06_lpt256 conv06_pm256m2 conv06_pm384m2

# 2. production ensemble and variants (LPT 1–7 + suppressed PM)
MODEL=all scripts/run_queue.sh config/fiducial.toml config/null.toml config/amp07.toml config/klo06.toml config/kuv06.toml
for c in fiducial null amp07 klo06 kuv06; do julia --project=. scripts/analyze.jl $c; done
julia --project=. scripts/diagnostics.jl fiducial

# 3. initial-condition controls (CPU)
CUDA_VISIBLE_DEVICES="" julia --project=. -t 32 scripts/controls.jl

# 4. figures and tables
$PY figures.py results figures
