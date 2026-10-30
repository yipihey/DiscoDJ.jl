#!/bin/bash
# Run all jobs of the given configs, relaunching the Julia process after a failure (e.g. GPU OOM);
# finished jobs are skipped.   usage: scripts/run_queue.sh config/a.toml [config/b.toml ...]
cd "$(dirname "$0")/.."
export JULIA_CUDA_HARD_MEMORY_LIMIT=${JULIA_CUDA_HARD_MEMORY_LIMIT:-44GiB}
for cfg in "$@"; do
  for attempt in 1 2 3 4 5; do
    julia --project=. scripts/run_jobs.jl "$cfg" ${MODEL:-all} && break
    echo "run_jobs failed for $cfg (attempt $attempt); relaunching"
  done
done
