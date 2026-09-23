#!/bin/bash
# laptop_placement.sh - PHASE C, the experiment CAL asked for.
#
# The referees' second fatal objection was that the paper attributes the cliff to
# efficiency cores without ever controlling placement: "there is no thread
# pinning, so the causal attribution to E-cores is not established." That is
# correct, and no amount of re-analysis fixes it. This holds the thread count
# fixed and moves ONLY where the threads run.
#
# THE DECISIVE COMPARISON is eight threads with the scheduler free to use
# E-cores, against eight threads masked to the performance cores' own logical
# processors. Same thread count, same work, same binary; the only difference is
# whether two of the threads land on Gracemont. If throughput recovers under the
# mask, the attribution is established rather than inferred.
#
# TOPOLOGY, MEASURED NOT ASSUMED. A pilot run pinned one thread to single
# logical processors and benchmarked compute-bound prefill: logical 0 gave 34.6
# tok/s and logical 2 gave 32.9, while logical 12 gave 25.5. Logical 0-11 are
# therefore the six hyperthreaded P-cores and 12-19 the eight E-cores, which
# Phase B re-establishes with ten repetitions on all twenty. The masks below
# follow from that, and phase B's output should be checked against them before
# these numbers are used.
#
#   0x555    bits 0,2,4,6,8,10  one logical per physical P-core (6 threads max)
#   0xFFF    bits 0-11          all P-core logical processors  (12 threads max)
#   0xFF000  bits 12-19         E-cores only                    (8 threads max)
#   0xFFFFF  bits 0-19          everything, stated explicitly
#
# The 0xFFFFF arm is a control on the instrument: if masking to every core
# differs from not masking at all, then --cpu-mask changes something besides
# placement and the other arms cannot be read as placement effects.
set -u

BIN=/c/llmpc/bin
MOD=/c/llmpc/models
WORK=/c/llmpc/sweep
OUT=$WORK/laptop_placement.jsonl
PY=python
mkdir -p "$MOD" "$WORK"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/place.log"; }

REPS=10
PROMPT=128
GEN=128
HF=https://huggingface.co

TARGETS="qwen0.5b|0.494|$HF/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
qwen1.5b|1.544|$HF/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# arm|mask|threads
ARMS="default|none|6,8,10,12,14,20
pall|0xFFFFF|8,20
ponly|0xFFF|6,8,10,12
pphys|0x555|4,6
eonly|0xFF000|2,4,6,8"

fetch(){ local u=$1 d=$2 k; for k in 1 2 3 4 5; do
  curl -sSL --retry 3 --retry-delay 5 -o "$d" "$u" 2>>"$WORK/place.log" && [ -s "$d" ] && return 0
  log "    fetch attempt $k failed"; sleep 10; done; return 1; }

log "=== laptop_placement (phase C) start ==="

while IFS='|' read -r tag params url; do
  [ -z "$tag" ] && continue
  f="$MOD/$tag.gguf"
  log "[$tag] download"
  fetch "$url" "$f" || { log "  DOWNLOAD FAILED"; rm -f "$f"; continue; }
  bytes=$(stat -c %s "$f")
  [ "$bytes" -lt 100000000 ] && { log "  BAD DOWNLOAD"; rm -f "$f"; continue; }
  sha=$(sha256sum "$f" | cut -d' ' -f1)
  log "[$tag] bytes=$bytes sha=${sha:0:12}"

  while IFS='|' read -r arm mask threads; do
    [ -z "$arm" ] && continue
    log "[$tag] arm=$arm mask=$mask threads=$threads"
    if [ "$mask" = "none" ]; then
      "$BIN/llama-bench.exe" -m "$f" -p "$PROMPT" -n "$GEN" -t "$threads" -r "$REPS" \
          -o json > "$WORK/place_${tag}_${arm}.json" 2>>"$WORK/place.log"
    else
      "$BIN/llama-bench.exe" -m "$f" -p "$PROMPT" -n "$GEN" -t "$threads" -r "$REPS" \
          -C "$mask" --cpu-strict 1 \
          -o json > "$WORK/place_${tag}_${arm}.json" 2>>"$WORK/place.log"
    fi
    "$PY" - "$WORK/place_${tag}_${arm}.json" "$tag" "$params" "$bytes" "$sha" \
           "$arm" "$mask" >> "$OUT" <<'PY'
import json, sys, statistics as st
f, tag, params, byts, sha, arm, mask = sys.argv[1:8]
try:
    rows = json.load(open(f))
except Exception as e:
    print(json.dumps({"tag": tag, "arm": arm, "error": str(e)})); raise SystemExit
for r in rows:
    s = r.get("samples_ts") or []
    print(json.dumps({
        "tag": tag, "params_B": float(params), "arm": arm, "cpu_mask": mask,
        "threads": r["n_threads"],
        "phase": "decode" if r.get("n_gen", 0) > 0 else "prefill",
        "bytes": int(byts), "sha256": sha, "build": r.get("build_commit"),
        "reps": len(s), "tok_s_mean": r["avg_ts"], "tok_s_stddev": r.get("stddev_ts"),
        "tok_s_median": st.median(s) if s else None,
        "tok_s_min": min(s) if s else None, "tok_s_max": max(s) if s else None,
        "samples_ts": s}))
PY
  done <<< "$ARMS"

  rm -f "$f"
  log "[$tag] done"
done <<< "$TARGETS"

log "=== DONE phase C: $(wc -l < "$OUT") rows ==="
