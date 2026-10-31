#!/bin/bash
# Experiment 3 IR members (run concurrently with tl_queue.sh; 20 GiB cap each, waits for 22 GiB free)
cd "$(dirname "$0")/.."
export EFTSC_SCRATCH=${EFTSC_SCRATCH:-/zpool/nvme/data/tabel_scratch/eft_shell_crossing}
export JULIA_CUDA_HARD_MEMORY_LIMIT=20GiB
wait_gpu() { while :; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ $((46068 - used)) -ge 22528 ] && return; sleep 60; done; }
for c in tl_ir020 tl_ir010 tl_ir015 tl_ir030; do wait_gpu; MODEL=all scripts/run_queue.sh config/$c.toml; done
echo "TL QUEUE3 DONE $(date)"
