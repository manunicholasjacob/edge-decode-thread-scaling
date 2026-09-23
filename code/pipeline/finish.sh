#!/bin/bash
# Single finisher: wait for the main sweep's own sentinel, then validate the
# CPU mask, then make up the models the network outage cost and re-run the
# headline model cleanly. Sequenced on sentinels written by each finishing job,
# never on process absence.
set -u
LOG=/c/llmpc/sweep/run.log
F=/c/llmpc/finish.log
say(){ echo "[$(date +%H:%M:%S)] $*" >> "$F"; }
say "finisher armed"
for i in $(seq 1 960); do                # up to 8 hours at 30s
  grep -q "=== DONE phases A and B" "$LOG" 2>/dev/null && break
  sleep 30
done
grep -q "=== DONE phases A and B" "$LOG" 2>/dev/null || { say "timed out waiting for sweep"; exit 1; }
for j in $(seq 1 40); do
  n=$(ps -W 2>/dev/null | grep -ci 'llama-bench'); n=${n:-0}
  [ "$n" -eq 0 ] && break
  sleep 15
done
say "sweep complete; validating cpu mask"
bash /c/llmpc/validate_mask.sh >> "$F" 2>&1
say "validation finished; starting makeup runs"
bash /c/llmpc/laptop_makeup.sh >> "$F" 2>&1
say "ALL FINISHED"
