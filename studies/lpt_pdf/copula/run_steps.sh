#!/bin/bash
# Steps 2-5 of the copula study (run only after step 1 passed): one command per (N, level, z).
# Every run writes results/steps/steps_N<N>_l<level>_z<z>.npz with the full config and seed.
# N = 256 at refinement 1 is out of reach of this machine (see run_step1.sh).
set -ex
for Z in 0 1; do
  for NL in "256 0" "128 0" "128 1" "64 0" "64 1"; do
    set -- $NL
    [ -f results/steps/steps_N$1_l$2_z$Z.npz ] || python3 analysis.py run $1 $2 $Z
  done
done
python3 analysis.py collect
