#!/bin/bash
# Step 1, then (only if every pass/fail validation test passed) steps 2-5.
set -e
cd "$(dirname "$0")"
./run_step1.sh
python3 - <<'PY'
import glob, json, sys
bad = [p for p in glob.glob("results/step1/[ab]_N*.json") if not json.load(open(p))["passed"]]
if not json.load(open("results/step1/d.json"))["passed_primary_mask"]:
    bad.append("d.json")
print("step-1 gate:", "FAILED " + ", ".join(bad) if bad else "all pass/fail tests passed", flush=True)
sys.exit(1 if bad else 0)
PY
./run_steps.sh
