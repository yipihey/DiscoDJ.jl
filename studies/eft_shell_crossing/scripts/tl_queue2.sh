#!/bin/bash
# runs after tl_queue.sh: the k_uv = 1.0 variant of the UV-divergent spectrum
cd "$(dirname "$0")/.."
until grep -q "TL QUEUE DONE" /zpool/nvme/data/tabel_scratch/eft_shell_crossing/tl_queue.log; do sleep 120; done
export EFTSC_SCRATCH=${EFTSC_SCRATCH:-/zpool/nvme/data/tabel_scratch/eft_shell_crossing}
export JULIA_CUDA_HARD_MEMORY_LIMIT=20GiB
while [ $((46068 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1))) -lt 22528 ]; do sleep 60; done
MODEL=all scripts/run_queue.sh config/tl_div10.toml
echo "TL QUEUE2 DONE $(date)"
