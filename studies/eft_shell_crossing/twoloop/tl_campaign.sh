#!/bin/bash
# Re-run the CPU measurements as GPU snapshots land, until every queue is done and nothing is left to measure.
cd "$(dirname "$0")"
D=${EFTSC_SCRATCH:-/zpool/nvme/data/tabel_scratch/eft_shell_crossing}
NJ=${1:-12}
while :; do
  ./tl_measure_all.sh $NJ > $D/tl_measure_batch.log 2>&1
  cat $D/tl_measure_batch.log >> $D/tl_measure_all.log
  n=$(head -1 $D/tl_measure_batch.log | awk '{print $1}')
  if grep -q "TL QUEUE DONE" $D/tl_queue.log && grep -q "TL QUEUE2 DONE" $D/tl_queue2.log \
     && grep -q "TL QUEUE3 DONE" $D/tl_queue3.log && [ "$n" = "0" ]; then echo "TL CAMPAIGN DONE $(date)" >> $D/tl_measure_all.log; break; fi
  [ "$n" = "0" ] && sleep 300
done
