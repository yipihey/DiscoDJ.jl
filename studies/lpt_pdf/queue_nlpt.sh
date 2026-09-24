#!/bin/bash
# 1LPT-4LPT on the same ICs: resolution series (3/2-rule de-aliased), an
# aliasing / higher-resolution check without de-aliasing, and the paired phases.
set -x
for N in 64 96 128 192 256; do python3 run.py nlpt $N; done
python3 run.py nlpt 256 --no-dealias
python3 run.py nlpt 384 --no-dealias
python3 run.py nlpt 256 --paired
