#!/bin/bash
# Stage 7: every tl_fit3 and closure fit, sequential, light CPU.  Logs in $EFTSC_SCRATCH/logs/.
cd "$(dirname "$0")"; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
P=/home/tabel/.local/share/anaconda3/bin/python3; L=$EFTSC_SCRATCH/logs
for pair in "tl_full IR" "tl_uv10 IR" "tl_uv06 IR" "tl_div IR" "tl_div10 IR" "tl_full tl_uv06" "tl_full tl_uv10" "tl_div tl_div10"; do
  set -- $pair; echo "fit3 $1 $2 $(date +%H:%M)"; $P tl_fit3.py $1 $2 pm,lpt3,lpt5,lpt7 0.12 > $L/fit3_$1_$2.log 2>&1 || echo "FAILED fit3 $1 $2"
done
for c in tl_full tl_uv10 tl_uv06 tl_div tl_div10; do for lam in 0.1 0.15 0.2 0.3; do
  echo "closure $c $lam $(date +%H:%M)"; $P tl_closure.py $c $lam pm > $L/closure_${c}_L$lam.log 2>&1 || echo "FAILED closure $c $lam"; mv ../results/tl_closure_$c.json ../results/tl_closure_${c}_L$lam.json 2>/dev/null
done; done
echo "STAGE7 FITS DONE $(date)"
