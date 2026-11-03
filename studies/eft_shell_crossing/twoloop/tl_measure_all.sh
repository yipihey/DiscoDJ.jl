#!/bin/bash
# Experiment 3 CPU measurements: every (config, model, seed, sign) whose inputs exist and whose output does not.
#   ./tl_measure_all.sh [njobs=12] [configs...]   (re-run as snapshots appear; FFTW single-threaded, 4 Julia threads each)
#   TL_SEEDS="1 2 3" restricts the seeds.  Campaign safety limit: at most 12 jobs. A PM or LPT member counts as ready only when its final member file exists:
#   snapshots are not written atomically, so a crash during the last snapshot must not pass as a complete member.
cd "$(dirname "$0")"
export EFTSC_SCRATCH=${EFTSC_SCRATCH:-/zpool/nvme/data/tabel_scratch/eft_shell_crossing}
export CUDA_VISIBLE_DEVICES="" TL_FILTER=${TL_FILTER:-sharp}
NJ=${1:-12}; [ "$NJ" -gt 12 ] && { echo "refusing more than 12 jobs (campaign safety rule)"; exit 2; }
shift; CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(tl_full tl_uv06 tl_uv10 tl_div tl_div10 tl_ir010 tl_ir015 tl_ir020 tl_ir030)
jobs=()
for cfg in "${CONFIGS[@]}"; do
  d=$EFTSC_SCRATCH/$cfg/snap; [ -d $d ] || continue
  case $cfg in tl_ir*) lam=$(python3 -c "print(int('${cfg:5}')/100)");; *) lam="0.1,0.15,0.2,0.3";; esac
  for s in ${TL_SEEDS:-1 2 3 4}; do for sg in +1 -1; do
    for m in pm lpt3 lpt5 lpt7; do
      out=$EFTSC_SCRATCH/$cfg/tl/stress_${TL_FILTER}_${m}_L${s}_s${sg}.h5
      [ -f $out ] && continue
      if [ $m = pm ]; then [ -f $EFTSC_SCRATCH/$cfg/pm_IR_L${s}_S0_s${sg}.h5 ] && [ -f $d/pm_IR_L${s}_S0_s${sg}_D1.000.h5 ] || continue
      else [ -f $EFTSC_SCRATCH/$cfg/lpt_IR_L${s}_S0_s${sg}.h5 ] && [ -f $d/lpt_IR_L${s}_S0_s${sg}_shapes.h5 ] || continue; fi
      jobs+=("$cfg $m $s ${sg#+} $lam")
    done
  done; done
  for s in 1 2 3 4; do
    out=$EFTSC_SCRATCH/$cfg/tl/ops_${TL_FILTER}_L${s}.h5
    [ $cfg = tl_full ] && { [ -f $out ] || jobs+=("$cfg ops $s 0 $lam"); }   # long-field operators: identical for every config (all agree below 0.3)
  done
done
echo "${#jobs[@]} jobs"
printf '%s\n' "${jobs[@]}" | xargs -P $NJ -L 1 bash -c '
  cfg=$0; m=$1; s=$2; sg=$3; lam=$4; mkdir -p $EFTSC_SCRATCH/$cfg/tl/log
  if [ $m = ops ]; then TL_LAMBDAS=$lam julia --project=.. -t 4 tl_measure.jl ops $cfg $s > $EFTSC_SCRATCH/$cfg/tl/log/ops_$s.log 2>&1
  else TL_LAMBDAS=$lam julia --project=.. -t 4 tl_measure.jl stress $cfg $m $s $sg > $EFTSC_SCRATCH/$cfg/tl/log/${m}_${s}_${sg}.log 2>&1; fi
  echo "done $cfg $m $s $sg ($?)"'
