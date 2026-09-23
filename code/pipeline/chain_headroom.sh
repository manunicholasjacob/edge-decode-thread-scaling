#!/bin/bash
# Run the headroom test after the x86 format-order control finishes. Watches
# fmt_chain.log for a sentinel that script writes, and logs elsewhere so the
# arming message cannot match what it waits on.
set -u
L=/c/llmpc/headroom_chain.log
say(){ echo "[$(date +%H:%M:%S)] $*" >> "$L"; }
say "armed, watching fmt_chain.log"
for i in $(seq 1 1600); do
  grep -q "FORMAT ORDER CONTROL FINISHED" /c/llmpc/fmt_chain.log 2>/dev/null && break
  sleep 30
done
grep -q "FORMAT ORDER CONTROL FINISHED" /c/llmpc/fmt_chain.log 2>/dev/null || { say "timed out"; exit 1; }
for j in $(seq 1 40); do
  n=$(ps -W 2>/dev/null | grep -ci 'llama-bench'); n=${n:-0}
  [ "$n" -eq 0 ] && break
  sleep 15
done
say "starting headroom test"
bash /c/llmpc/headroom_test.sh >> "$L" 2>&1
say "HEADROOM TEST FINISHED"
