#!/bin/bash
# sigma(M) for the UV-cutoff and UV-divergent spectra, after the tl_ir020 run; 4 epochs, 32 threads; finished snapshots are skipped.
cd "$(dirname "$0")"
D=$EFTSC_SCRATCH
while kill -0 $(cat $D/sigmaM.pid) 2>/dev/null; do sleep 15; done
echo $$ > $D/sigmaM.pid
for c in tl_uv06 tl_uv10 tl_div tl_div10; do
  CUDA_VISIBLE_DEVICES="" julia --project=.. -t 32 sigmaM.jl $c 1,2,3,4 1,-1 0.3,0.5,0.8,1.0
done
echo "SIGMAM CHAIN DONE $(date)"
