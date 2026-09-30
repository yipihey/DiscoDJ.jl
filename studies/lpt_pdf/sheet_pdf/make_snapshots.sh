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
if [ ! -f $S/psi_nbody_N256_z0_paired.npy ]; then
  B=$(python3 snapshots.py fphi 256 --paired)
  $JL djn_snapshots.jl $B 256 $S 1lpt,2lpt,3lpt,4lpt,nbody _paired
fi
