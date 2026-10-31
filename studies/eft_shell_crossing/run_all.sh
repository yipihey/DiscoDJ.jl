#!/bin/bash
# Reproduce the EFT / shell-crossing study end to end.
#
#   EFTSC_SCRATCH=/path/to/scratch ./run_all.sh            (≈ 12 GPU-hours on one RTX A6000)
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
MODEL=all scripts/run_queue.sh config/fiducial.toml config/null.toml config/amp07.toml config/kuv06.toml
for c in fiducial null amp07 kuv06; do julia --project=. scripts/analyze.jl $c; done
julia --project=. scripts/diagnostics.jl fiducial
julia --project=. scripts/spheres_from_modes.jl fiducial
julia --project=. scripts/c4_convergence.jl conv_pm256m2bf conv_pm384m2bf conv_pm512m2lean > results/c4_convergence.txt

# 3. RG running (7 sharp splits sharing one UV member per seed) with counts in spheres and redshift space
MODEL=all scripts/run_queue.sh config/rg_ks20.toml config/rg_ks12.toml config/rg_ks15.toml config/rg_ks25.toml \
          config/rg_ks33.toml config/rg_ks50.toml config/rg_ks70.toml
RG="rg_ks12 rg_ks15 rg_ks20 rg_ks25 rg_ks33 rg_ks50 rg_ks70"
for c in $RG; do julia --project=. scripts/analyze.jl $c; done
julia --project=. scripts/cis_analysis.jl $RG
julia --project=. scripts/rsd_analysis.jl $RG

# 4. experiment 2: kinetic multistream stress; small-scale P(k) convergence
MODEL=all scripts/run_queue.sh config/stress.toml
julia --project=. scripts/stress_analysis.jl stress
MODEL=pm scripts/run_queue.sh config/pkconv_pm256bf.toml config/pkconv_pm384bf.toml config/pkconv_pm512lean.toml
julia --project=. scripts/pk_convergence.jl pkconv_pm256bf pkconv_pm384bf pkconv_pm512lean

# 5. initial-condition controls (CPU), CHS reconstruction and band-edge control
CUDA_VISIBLE_DEVICES="" julia --project=. -t 32 scripts/controls.jl
$PY chs/chs_fig3.py; $PY chs/band_edge.py          # needs camb

# 5b. plane-wave (ε → 0) counterterms, §14: time-step ladder, k_split ladder, long-background scan, A/B kernel
export JULIA_CUDA_HARD_MEMORY_LIMIT=${JULIA_CUDA_HARD_MEMORY_LIMIT:-20GiB}     # leave room for other GPU users
MODEL=pm scripts/run_queue.sh config/steps_pm256m2bf_n{8,16,32,64,400}.toml config/steps_pm256m2bf_n{16,32,100}_di15o7.toml
julia --project=. scripts/step_convergence.jl steps_pm256m2bf_n400 steps_pm256m2bf_n8 steps_pm256m2bf_n16 steps_pm256m2bf_n32 \
      steps_pm256m2bf_n64 conv_pm256m2bf steps_pm256m2bf_n16_di15o7 steps_pm256m2bf_n32_di15o7 steps_pm256m2bf_n100_di15o7
PW="prod_ks12 prod_ks15 prod_ks20 prod_ks25 prod_ks33 prod_ks50 prod_ks70"
for c in $PW prod_ks20_bg05 prod_ks20_bg10 prod_ks20_ab; do MODEL=all scripts/run_queue.sh config/$c.toml; done
MODEL=pm scripts/run_queue.sh config/prod_ks50_ir.toml             # bare-wave PM diagnostic (§14.2)
julia --project=. scripts/pw_production.jl $PW prod_ks20_bg05 prod_ks20_bg10
julia --project=. scripts/pw_robustness.jl $PW prod_ks20_bg05 prod_ks20_bg10
julia --project=. scripts/ab_kernel.jl prod_ks20 prod_ks20_ab

# 6. figures and tables
EFTSC_FIG_CONFIG=fiducial $PY figures.py results figures
$PY figures_rg.py results figures
$PY figures_pw.py results figures
