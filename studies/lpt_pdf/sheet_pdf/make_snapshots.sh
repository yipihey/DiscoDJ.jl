#!/bin/bash
# z = 0 displacement snapshots for the sheet-estimator PDF study, all from DiscoDJNative, on the
# study-1 initial conditions (same fixed-amplitude master phases; --paired = theta + pi partner).
# The fixed-phase 1LPT/2LPT/4LPT/N-body snapshots at N = 64, 128, 256 are shared with the copula
# study (copula/make_snapshots.sh); this adds 3LPT and the paired-phase set at N = 256.
set -ex
cd "$(dirname "$0")/../copula"
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/opt/jdepot} JULIA_PKG_SERVER=${JULIA_PKG_SERVER:-} DJN_ZLIST=0
JL="${JULIA:-/opt/jl/bin/julia} -t auto --project=."
S=../_scratch/copula
for N in 64 128 256; do
  [ -f $S/psi_3lpt_N${N}_z0.npy ] && continue
  B=$(python3 snapshots.py fphi $N)
  $JL djn_snapshots.jl $B $N $S 3lpt
done
# paired set at N = 256: LPT orders and N-body in separate processes, lean nLPT kernels (15 GB)
B=$(python3 snapshots.py fphi 256 --paired)
[ -f $S/psi_3lpt_N256_z0_paired.npy ] || DJN_MODE=lean $JL djn_snapshots.jl $B 256 $S 1lpt,2lpt,3lpt _paired
[ -f $S/psi_4lpt_N256_z0_paired.npy ] || DJN_MODE=lean $JL djn_snapshots.jl $B 256 $S 4lpt _paired
[ -f $S/psi_nbody_N256_z0_paired.npy ] || DJN_MODE=lean $JL djn_snapshots.jl $B 256 $S nbody _paired
