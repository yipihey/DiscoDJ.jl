#!/bin/zsh
# Staged launcher used for the production campaign (documented here for reproducibility).
# stage 2: after the cheap N=64 ensembles finish → R_f=22 (4 workers) + L1000 N=256 (4 workers)
# stage 3: after the 5LPT and R14 ensembles finish → 6LPT (2 workers) + L2000 N=256 (4 workers) + FEM N=256 (1 worker)
cd "$(dirname "$0")/.."
J="julia -t 4 --project=."
wait_for() { while [ "$(ls $1 2>/dev/null | wc -l | tr -d ' ')" -lt "$2" ]; do sleep 60; done; }
stage=$1
if [ "$stage" = "2" ]; then
  wait_for "data/L1000_N64_R18/part_*-400.jld2" 2; wait_for "data/L500_N64_R18/part_*-400.jld2" 2
  mkdir -p data/L1000_N128_R22 data/L1000_N256_R18
  for r in "gaussian 1 75" "gaussian 76 150" "fixed 1 75" "fixed 76 150"; do set -A a ${=r}; nohup $J bin/run_ensemble.jl config/L1000_N128_R22.toml $a[1] $a[2] $a[3] > data/L1000_N128_R22/stdout_$a[1]_$a[2].txt 2>&1 & done
  for r in "gaussian 1 30" "gaussian 31 60" "fixed 1 30" "fixed 31 60"; do set -A a ${=r}; nohup $J bin/run_ensemble.jl config/L1000_N256_R18.toml $a[1] $a[2] $a[3] > data/L1000_N256_R18/stdout_$a[1]_$a[2].txt 2>&1 & done
  echo "stage 2 launched $(date)"
elif [ "$stage" = "3" ]; then
  wait_for "data/L1000_N128_R18_5LPT/part_*.jld2" 4; wait_for "data/L1000_N128_R14/part_*.jld2" 4
  mkdir -p data/L1000_N128_R18_6LPT data/L2000_N256_R18 data/FEM_L500_N256
  for r in "gaussian 1 50" "fixed 1 50"; do set -A a ${=r}; nohup $J bin/run_ensemble.jl config/L1000_N128_R18_6LPT.toml $a[1] $a[2] $a[3] > data/L1000_N128_R18_6LPT/stdout_$a[1]_$a[2].txt 2>&1 & done
  for r in "gaussian 1 30" "gaussian 31 60" "fixed 1 30" "fixed 31 60"; do set -A a ${=r}; nohup $J bin/run_ensemble.jl config/L2000_N256_R18.toml $a[1] $a[2] $a[3] > data/L2000_N256_R18/stdout_$a[1]_$a[2].txt 2>&1 & done
  nohup $J bin/run_fem.jl config/FEM_L500_N256.toml 1 2 > data/FEM_L500_N256/stdout.txt 2>&1 &
  echo "stage 3 launched $(date)"
fi
