#!/bin/bash
# All displacement snapshots for the copula study (N-body PM runs + LPT orders + ZA band seeds).
set -ex
for N in 64 128 256; do python3 snapshots.py lpt $N; done
for S in 101 102 103 104 105 106 107 108; do python3 snapshots.py za 128 --seed $S; done
for N in 64 128 256; do python3 snapshots.py nbody $N; done
