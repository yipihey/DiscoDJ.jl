#!/bin/bash
# All displacement snapshots for the copula study, computed with DiscoDJNative (Julia):
# nLPT (compute_core_exact / compute_core) and the port of DISCO-DJ run_nbody (BullFrog).
# The IC potential (fixed amplitudes, top-hat filter, shared phases) is written by snapshots.py.
set -ex
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/opt/jdepot} JULIA_PKG_SERVER=${JULIA_PKG_SERVER:-}
JL="${JULIA:-/opt/jl/bin/julia} -t auto --project=$(dirname "$0")"
S=../_scratch/copula
for N in 64 128 256; do
  B=$(python3 snapshots.py fphi $N)
  $JL djn_snapshots.jl $B $N $S 1lpt,2lpt,4lpt,nbody
done
for SEED in 101 102 103 104 105 106 107 108; do          # Zel'dovich band realisations (test 1d)
  B=$(python3 snapshots.py fphi 128 --seed $SEED)
  $JL djn_snapshots.jl $B 128 $S 1lpt _seed$SEED
done
