#!/bin/bash
# Stage 4: a CLEAN per-core scan, now that the mask is proven to bind.
#
# The first version of this script fired instantly because it grepped the same
# log it wrote its own arming message into, and that message quoted the sentinel
# it was waiting for. It then started a benchmark alongside the makeup runs.
# Same self-matching trap as `ps | grep -c` counting its own grep.
#
# Fixed by watching a DIFFERENT file: laptop_artifact_control.sh writes its
# completion line into sweep/artifact.log, and nothing here ever writes there.
set -u
F=/c/llmpc/finish.log
WATCH=/c/llmpc/sweep/artifact.log
SENTINEL="ARTIFACT CONTROL DONE"
say(){ echo "[$(date +%H:%M:%S)] [stage4] $*" >> "$F"; }
say "armed, watching sweep/artifact.log"
for i in $(seq 1 1200); do
  grep -q "$SENTINEL" "$WATCH" 2>/dev/null && break
  sleep 30
done
grep -q "$SENTINEL" "$WATCH" 2>/dev/null || { say "timed out"; exit 1; }
for j in $(seq 1 40); do
  n=$(ps -W 2>/dev/null | grep -ci 'llama-bench'); n=${n:-0}
  [ "$n" -eq 0 ] && break
  sleep 15
done
say "starting clean per-core scan"
bash /c/llmpc/coreid_clean.sh >> "$F" 2>&1
say "CORE SCAN COMPLETE"
