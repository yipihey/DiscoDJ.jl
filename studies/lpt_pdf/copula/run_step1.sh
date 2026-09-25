#!/bin/bash
# Step 1 (validation) of the copula study: one command per test / resolution / z / refinement.
# Sheet products come from DiscoDJNative (Julia) and are deleted after each (N, z, level) group.
# Every call writes results/step1/<test>_*.json with the full config and seeds.
set -ex
until grep -q "nbody N=256 z=0 saved" ../_scratch/copula/make_snapshots.log; do sleep 30; done
python3 validate.py d                                   # Zel'dovich (own snapshots, geometry only)
for N in 64 128 256; do
  for LV in 0 1; do
    for Z in 24 1 0; do
      [ "$Z" = "24" ] && python3 validate.py a $N $LV
      if [ "$Z" = "0" ] && { [ "$LV" = "0" ] || [ "$N" = "128" ]; }; then python3 validate.py b $N $LV; fi
      if [ "$LV" = "0" ]; then python3 validate.py c $N 0 $Z; else python3 validate.py c $N 1 $Z 11; fi
      python3 validate.py cleanup $N $Z $LV
    done
  done
done
python3 collect.py
