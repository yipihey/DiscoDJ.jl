#!/bin/bash
# Test 1d band: independent fixed-amplitude Zel'dovich realisations (seeds from config test_d.extra_seeds).
# Per seed: IC potential -> DiscoDJNative 1LPT snapshot -> sheet products -> cached quantiles
# (_scratch/copula/zaq_*.json); snapshot and products are then deleted to bound disk. Idempotent.
set -ex
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/opt/jdepot} JULIA_PKG_SERVER=${JULIA_PKG_SERVER:-}
JL="${JULIA:-/opt/jl/bin/julia} -t auto --project=$(dirname "$0")"
S=../_scratch/copula
N=$(python3 -c 'from common import CFG; print(CFG["test_d"]["N_band"])')
LV=$(python3 -c 'from common import CFG; print(CFG["test_d"]["refine_band"])')
for SEED in $(python3 -c 'from common import CFG; print(*CFG["test_d"]["extra_seeds"])'); do
  [ -f $S/zaq_N${N}_l${LV}_z0_seed$SEED.json ] && [ -f $S/zaq_N${N}_l${LV}_z1_seed$SEED.json ] && continue
  if [ ! -f $S/psi_1lpt_N${N}_z0_seed$SEED.npy ]; then
    B=$(python3 snapshots.py fphi $N --seed $SEED)
    $JL djn_snapshots.jl $B $N $S 1lpt _seed$SEED
    rm -f ${B}_re.npy ${B}_im.npy
  fi
  python3 validate.py dseed $SEED
  rm -f $S/psi_1lpt_N${N}_z*_seed$SEED.npy
done
