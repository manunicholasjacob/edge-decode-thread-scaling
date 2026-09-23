#!/bin/bash
# Run the x86 format-order control once the size sweep writes its own sentinel.
# Watches sweep/randsize.log and logs elsewhere, so the arming message cannot
# match the sentinel it waits on.
set -u
L=/c/llmpc/fmt_chain.log
say(){ echo "[$(date +%H:%M:%S)] $*" >> "$L"; }
say "armed, watching sweep/randsize.log"
for i in $(seq 1 1400); do
  grep -q "RANDOMIZED SIZE SWEEP DONE" /c/llmpc/sweep/randsize.log 2>/dev/null && break
  sleep 30
done
grep -q "RANDOMIZED SIZE SWEEP DONE" /c/llmpc/sweep/randsize.log 2>/dev/null || { say "timed out"; exit 1; }
for j in $(seq 1 40); do
  n=$(ps -W 2>/dev/null | grep -ci 'llama-bench'); n=${n:-0}
  [ "$n" -eq 0 ] && break
  sleep 15
done
say "starting x86 format order control"
bash /c/llmpc/x86_format_order.sh >> "$L" 2>&1
say "FORMAT ORDER CONTROL FINISHED"
