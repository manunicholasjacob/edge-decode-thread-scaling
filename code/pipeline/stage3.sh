#!/bin/bash
set -u
F=/c/llmpc/finish.log
say(){ echo "[$(date +%H:%M:%S)] [stage3] $*" >> "$F"; }
say "armed, waiting for MAKEUP DONE"
for i in $(seq 1 960); do
  grep -q "=== MAKEUP DONE" /c/llmpc/sweep/makeup.log 2>/dev/null && break
  sleep 30
done
grep -q "=== MAKEUP DONE" /c/llmpc/sweep/makeup.log 2>/dev/null || { say "timed out"; exit 1; }
for j in $(seq 1 40); do
  n=$(ps -W 2>/dev/null | grep -ci 'llama-bench'); n=${n:-0}
  [ "$n" -eq 0 ] && break
  sleep 15
done
say "starting artifact control"
bash /c/llmpc/laptop_artifact_control.sh >> "$F" 2>&1
say "ARTIFACT CONTROL FINISHED"
