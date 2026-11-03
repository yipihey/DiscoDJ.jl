#!/bin/bash
# nLPT fixed-mass sigmas on the stored shapes: 8 jobs of 8 threads, skipping finished outputs.  Run from studies/eft_shell_crossing.
export CUDA_VISIBLE_DEVICES=""
{
for m in lpt1 lpt2 lpt3 lpt5 lpt7; do for s in 1 2 3 4; do for g in 1 -1; do echo "tl_full $s $g 0.2,0.3,0.5,0.8,1.0 $m"; done; done; done
for m in lpt1 lpt3 lpt7; do for s in 1 2 3 4; do for g in 1 -1; do echo "tl_div $s $g 0.3,0.5,0.8,1.0 $m"; done; done; done
} | xargs -P 8 -L 1 julia --project=. -t 8 twoloop/sigmaM.jl
echo QUEUE_DONE
