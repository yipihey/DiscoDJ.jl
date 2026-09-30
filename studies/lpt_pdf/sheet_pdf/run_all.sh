#!/bin/bash
# Whole sheet-PDF study: snapshots, both estimators, statistics, tables and figures (resumable).
set -ex
cd "$(dirname "$0")"
./make_snapshots.sh                      # needs ~13 GB for the paired N = 256 set: run alone
python3 run_sheet_pdf.py deposit         # CIC + exact sheet for every snapshot (skips existing)
python3 run_sheet_pdf.py stats
python3 analyze_sheet.py
