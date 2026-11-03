#!/bin/bash
# Experiment 3 (two-loop stress) queue (shared GPU: waits for ≥ 22 GiB free, caps this process at 20 GiB).
cd "$(dirname "$0")/.."
export EFTSC_SCRATCH=${EFTSC_SCRATCH:-/zpool/nvme/data/tabel_scratch/eft_shell_crossing}
export JULIA_CUDA_HARD_MEMORY_LIMIT=20GiB
wait_gpu() {
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ $((46068 - used)) -ge 22528 ] && return
    sleep 60
  done
}
run() { wait_gpu; MODEL=$1 scripts/run_queue.sh "${@:2}"; }
run all config/tl_full.toml
run all config/tl_uv06.toml
run all config/tl_div.toml
run all config/tl_uv10.toml
echo "TL QUEUE DONE $(date)"
