#!/bin/bash
# Step 1 (validation) of the copula study: one command per test / resolution / z / refinement.
# Snapshots: DiscoDJNative (make_snapshots.sh). Sheet products come from DiscoDJNative (Julia) and are deleted after each (N, z, level) group.
# Every call writes results/step1/<test>_*.json with the full config and seeds.
set -ex
./run_test_d_band.sh                                    # Zel'dovich band realisations (cached quantiles)
python3 validate.py d                                   # Zel'dovich (own snapshots, geometry only)
rm -f ../_scratch/copula/prod_geom_1lpt_N*_z*_l1.h5      # test-d products (quantiles are cached)
for N in 64 128 256; do
  for LV in 0 1; do
    # N = 256 at refinement 1 (512^3 = 1.3e8 elements) needs ~20 GB of product files and >15 GB RAM
    # for tests a-c; it is out of reach of this machine, so refinement is tracked at N = 64, 128 only.
    [ "$N" = "256" ] && [ "$LV" = "1" ] && continue
    for Z in 24 1 0; do
      [ "$Z" = "24" ] && python3 validate.py a $N $LV
      if [ "$Z" = "0" ] && { [ "$LV" = "0" ] || [ "$N" = "128" ]; }; then python3 validate.py b $N $LV; fi
      if [ "$LV" = "0" ]; then python3 validate.py c $N 0 $Z; else python3 validate.py c $N 1 $Z 11; fi
      python3 validate.py cleanup $N $Z $LV
    done
  done
done
python3 collect.py
