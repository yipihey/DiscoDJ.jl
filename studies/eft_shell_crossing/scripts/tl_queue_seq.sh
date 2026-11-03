#!/bin/bash
# Experiment 3 GPU members, one process at a time (replaces tl_queue.sh, tl_queue2.sh, tl_queue3.sh, which assume concurrency;
# tl_queue2.sh's wait on "TL QUEUE DONE" would never fire after a crash).  Finished jobs are skipped by run_jobs.jl.
# Stops before the next config if $EFTSC_SCRATCH/STOP_GPU exists.  20 GiB cap; waits for 22 GiB free on the shared card.
cd "$(dirname "$0")/.."
export EFTSC_SCRATCH=${EFTSC_SCRATCH:-/zpool/nvme/data/tabel_scratch/eft_shell_crossing}
export JULIA_CUDA_HARD_MEMORY_LIMIT=20GiB
D=$EFTSC_SCRATCH
wait_gpu() { while :; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ $((46068 - used)) -ge 22528 ] && return; sleep 60; done; }
for c in ${TL_CONFIGS:-tl_ir020 tl_div tl_uv10 tl_div10 tl_ir010 tl_ir015 tl_ir030}; do
  [ -f $D/STOP_GPU ] && { echo "STOP_GPU present before $c $(date)"; exit 0; }
  case $c in tl_ir*) log=$D/tl_queue3.log;; tl_div10) log=$D/tl_queue2.log;; *) log=$D/tl_queue.log;; esac
  wait_gpu
  echo "TL SEQ START $c $(date)" | tee -a $log
  MODEL=all scripts/run_queue.sh config/$c.toml >> $log 2>&1
  n=$(find $D/$c -maxdepth 1 -name '*_IR_L*.h5' | wc -l)
  echo "TL SEQ END $c members=$n/16 relaunches=$(grep -c 'run_jobs failed for config/'$c $log) $(date)" | tee -a $log
done
echo "TL SEQ ALL DONE $(date)"
