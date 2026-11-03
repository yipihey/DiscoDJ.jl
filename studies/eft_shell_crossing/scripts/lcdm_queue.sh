#!/bin/bash
# ΛCDM counterterm calibration queue (shared GPU: waits for ≥ 22 GiB free, caps this process at 20 GiB).
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
run all config/lcdm_full.toml
run lpt config/lcdm_cut010.toml config/lcdm_cut015.toml config/lcdm_cut020.toml config/lcdm_cut030.toml config/lcdm_cut050.toml
run all config/lcdm_cut100.toml
run all config/lcdm_s8lo.toml config/lcdm_s8hi.toml
echo "LCDM QUEUE DONE $(date)"
