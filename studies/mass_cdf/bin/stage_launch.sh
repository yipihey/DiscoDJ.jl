#!/bin/zsh
# Launch commands used for the production campaign (kept for reproducibility; run from studies/mass_cdf).
#   zsh bin/stage_launch.sh 1|2|3
# Each worker: julia -t 4 --project=. bin/run_ensemble.jl <config> <ic_type> <seed_lo> <seed_hi>
cd "$(dirname "$0")/.."
launch_all() {   # arguments: "<ensemble> <ic> <lo> <hi>" ...
  for r in "$@"; do set -A a ${=r}; mkdir -p data/$a[1]
    nohup julia -t 4 --project=. bin/run_ensemble.jl config/$a[1].toml $a[2] $a[3] $a[4] > data/$a[1]/stdout_$a[2]_$a[3].txt 2>&1 &
  done
}
case "$1" in
  1) launch_all "L1000_N128_R18 gaussian 1 100" "L1000_N128_R18 gaussian 101 200" "L1000_N128_R18 gaussian 201 300" "L1000_N128_R18 gaussian 301 400" \
                "L1000_N128_R18 fixed 1 100" "L1000_N128_R18 fixed 101 200" \
                "L1000_N128_R18_5LPT gaussian 1 100" "L1000_N128_R18_5LPT gaussian 101 200" "L1000_N128_R18_5LPT fixed 1 100" "L1000_N128_R18_5LPT fixed 101 200" \
                "L1000_N128_R14 gaussian 1 75" "L1000_N128_R14 gaussian 76 150" "L1000_N128_R14 fixed 1 75" "L1000_N128_R14 fixed 76 150" \
                "L1000_N64_R18 gaussian 1 400" "L1000_N64_R18 fixed 1 400" "L500_N64_R18 gaussian 1 400" "L500_N64_R18 fixed 1 400"
     mkdir -p data/FEM_L500_N64 data/FEM_L500_N128 data/FEM_L500_N64_steps100
     nohup julia -t 4 --project=. bin/run_fem.jl config/FEM_L500_N64.toml 1 10 > data/FEM_L500_N64/stdout.txt 2>&1 &
     nohup julia -t 4 --project=. bin/run_fem.jl config/FEM_L500_N128.toml 1 3 > data/FEM_L500_N128/stdout.txt 2>&1 &
     nohup julia -t 4 --project=. bin/run_fem.jl config/FEM_L500_N64_steps.toml 1 1 > data/FEM_L500_N64_steps100/stdout.txt 2>&1 & ;;
  2) launch_all "L1000_N128_R22 gaussian 1 75" "L1000_N128_R22 gaussian 76 150" "L1000_N128_R22 fixed 1 75" "L1000_N128_R22 fixed 76 150" \
                "L1000_N256_R18 gaussian 1 30" "L1000_N256_R18 gaussian 31 60" "L1000_N256_R18 fixed 1 30" "L1000_N256_R18 fixed 31 60" ;;
  3) launch_all "L1000_N128_R18_6LPT gaussian 1 50" "L1000_N128_R18_6LPT fixed 1 50" \
                "L2000_N256_R18 gaussian 1 30" "L2000_N256_R18 gaussian 31 60" "L2000_N256_R18 fixed 1 30" "L2000_N256_R18 fixed 31 60"
     mkdir -p data/FEM_L500_N256
     nohup julia -t 4 --project=. bin/run_fem.jl config/FEM_L500_N256.toml 1 1 > data/FEM_L500_N256/stdout.txt 2>&1 & ;;
esac
echo "stage $1 launched $(date)"
