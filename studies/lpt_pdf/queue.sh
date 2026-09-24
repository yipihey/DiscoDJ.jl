#!/bin/bash
# Full run queue for the study (sequential; each run uses all cores).
set -x
for N in 64 96 128 192 256 384 512; do python3 run.py lpt $N; done
for N in 64 96 128; do python3 run.py nbody $N; done
for S in 25 50 200; do python3 run.py nbody 128 --steps $S; done
python3 run.py nbody 192
python3 run.py nbody 256
python3 run.py nbody 128 --mesh 1
python3 run.py nbody 128 --mesh 3
python3 run.py nbody 128 --ai 0.02
python3 run.py nbody 128 --ai 0.0833333
python3 run.py nbody 256 --steps 50
python3 run.py lpt 256 --paired
python3 run.py lpt 512 --paired
python3 run.py nbody 256 --paired
