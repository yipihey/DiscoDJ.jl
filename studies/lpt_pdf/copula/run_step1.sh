#!/bin/bash
# Step 1 (validation) of the copula study.  One command per test / resolution / z;
# every call writes results/step1/<test>_*.json with the full config and seeds.
set -ex
until grep -q "nbody N=256 z=0 saved" ../_scratch/copula/make_snapshots.log; do sleep 30; done
for N in 64 128 256; do for LV in 0 1; do python3 validate.py a $N $LV; done; done
for N in 64 128 256; do python3 validate.py b $N 0; done
python3 validate.py b 128 1
for Z in 24 1 0; do
  for N in 64 128 256; do
    python3 validate.py c $N 0 $Z            # tie seeds 11,12,13 -> seed floor
    python3 validate.py c $N 1 $Z 11         # refinement level 1 (refinement floor vs level 0)
  done
done
