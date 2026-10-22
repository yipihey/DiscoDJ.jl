#!/bin/zsh
# Wait for the remaining production ensembles and run all analyses (used at the end of the campaign).
setopt glob_subst
cd "$(dirname "$0")/.."
count() { { ls $1 } 2>/dev/null | wc -l | tr -d " " }
wait_n() { while [ "$(count $1)" -lt "$2" ]; do sleep 60; done; echo "$(date) ready: $1"; }
A="julia --project=. bin/analyze.jl"
wait_n "data/L1000_N128_R22/part_*.jld2" 4;      julia --project=. bin/analyze.jl L1000_N128_R22      > results/analyze_L1000_N128_R22.log 2>&1
wait_n "data/FEM_L500_N128/part_fem_*.jld2" 1
wait_n "data/L1000_N128_R18_6LPT/part_*.jld2" 2; julia --project=. bin/analyze.jl L1000_N128_R18_6LPT > results/analyze_L1000_N128_R18_6LPT.log 2>&1
wait_n "data/L1000_N256_R18/part_*.jld2" 4;      julia --project=. bin/analyze.jl L1000_N256_R18      > results/analyze_L1000_N256_R18.log 2>&1
wait_n "data/L2000_N256_R18/part_*.jld2" 4;      julia --project=. bin/analyze.jl L2000_N256_R18      > results/analyze_L2000_N256_R18.log 2>&1
wait_n "data/FEM_L500_N256/part_fem_*.jld2" 1
julia --project=. bin/analyze_fem.jl FEM_L500_N64 FEM_L500_N128 FEM_L500_N256 > results/analyze_fem.log 2>&1
julia --project=. bin/analyze_convergence.jl > results/analyze_convergence.log 2>&1
julia --project=. bin/make_figures.jl L1000_N128_R18 L1000_N64_R18 L1000_N256_R18 L500_N64_R18 L2000_N256_R18 L1000_N128_R14 L1000_N128_R22 > results/make_figures.log 2>&1
echo "$(date) finalize done"
