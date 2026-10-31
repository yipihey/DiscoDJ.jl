#!/bin/bash
# Re-run the CPU measurements as GPU snapshots land, until every queue is done and nothing is left to measure.
cd "$(dirname "$0")"
D=${EFTSC_SCRATCH:-/zpool/nvme/data/tabel_scratch/eft_shell_crossing}
while pgrep -f "tl_measure_all.sh" > /dev/null; do sleep 60; done          # let a running batch finish
while :; do
  n=$(./tl_measure_all.sh 22 | tee -a $D/tl_measure_all.log | head -1 | awk '{print $1}')
  if grep -q "TL QUEUE DONE" $D/tl_queue.log && grep -q "TL QUEUE2 DONE" $D/tl_queue2.log \
     && grep -q "TL QUEUE3 DONE" $D/tl_queue3.log && [ "$n" = "0" ]; then echo "TL CAMPAIGN DONE $(date)" >> $D/tl_measure_all.log; break; fi
  sleep 300
done
