#!/bin/bash
# randomized_sweep_sizes.sh - put the REST of the size sweep through the
# protocol that the 0.5B model has already been through.
#
# WHY THIS IS NOT OPTIONAL. The randomised protocol changed the 0.5B answer
# completely: a peak at sixteen threads rather than eight, a 20.8% median fall
# rather than 35.1%, and the real effect turning out to be a collapse in
# RELIABILITY (coefficient of variation 2.7% at the peak against 31% at twenty
# threads) rather than in throughput. The paper's size-scaling law, that the
# cliff deepens as the model shrinks, still rests entirely on the sequential
# ascending sweeps whose confound this protocol exists to remove. Reporting a
# law derived from four confounded curves and one clean one would be worse than
# reporting nothing.
#
# So every model gets the same treatment: five passes, each visiting the thread
# counts in a different seeded permutation, one llama-bench invocation per cell
# with a cooldown between, recording pass and position so drift can be measured
# rather than assumed away.
#
# The 0.5B model is excluded because it is already done, and its data lives in
# randomized_sweep.jsonl. The 7B ships as two shards.
#
# This is long. At the larger models a cell is a minute and there are 75 of
# them per model. Expect several hours, and do not run anything else on the
# machine while it works.
set -u

BIN=/c/llmpc/bin
MOD=/c/llmpc/models
WORK=/c/llmpc/sweep
OUT=$WORK/randomized_sweep_7b.jsonl
mkdir -p "$MOD" "$WORK"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/rand7b.log"; }

THREADS="1 2 3 4 5 6 7 8 9 10 12 14 16 18 20"
PASSES=5
REPS=3
COOL=20
HF=https://huggingface.co

TARGETS="qwen7b|7.616|$HF/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

fetch(){ local u=$1 d=$2 k
  for k in $(seq 1 20); do
    curl -sSL --retry 3 --retry-delay 10 --connect-timeout 30 -o "$d" "$u" \
      2>>"$WORK/rand7b.log" && [ -s "$d" ] && return 0
    log "    fetch attempt $k failed"; sleep 30
  done; return 1; }

log "=== randomized 7B sweep: $PASSES passes, r=$REPS, ${COOL}s cooldown ==="

while IFS='|' read -r tag params url; do
  [ -z "$tag" ] && continue
  f="$MOD/rz_$tag.gguf"
  d="$MOD/rz7b"
  S1="$d/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
  S2="$d/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
  mkdir -p "$d"

  if [ "$tag" = "qwen7b" ] && [ -s "$S1" ] && [ -s "$S2" ]; then
    log "[$tag] both shards already on disk, skipping download"
  else
    log "[$tag] download"
    fetch "$url" "$f" || { log "  DOWNLOAD FAILED"; rm -f "$f"; continue; }
  fi

  if [ "$tag" = "qwen7b" ]; then
    [ -s "$f" ] && mv "$f" "$S1"
    if [ ! -s "$S2" ]; then
      fetch "$HF/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf" "$S2" \
        || { log "  SHARD 2 FAILED"; rm -rf "$d"; continue; }
    fi
    f="$S1"
    bytes=$(stat -c %s "$d"/*.gguf | awk '{s+=$1} END{print s}')
  else
    bytes=$(stat -c %s "$f")
  fi
  if [ "$bytes" -lt 100000000 ]; then
    log "  BAD DOWNLOAD ($bytes bytes)"; rm -rf "$f" "$MOD/rz7b"; continue
  fi
  sha=$(sha256sum "$f" | cut -d' ' -f1)
  log "[$tag] bytes=$bytes sha=${sha:0:12}"

  for pass in $(seq 1 "$PASSES"); do
    order=$(python -c "
import random
random.seed(1000 + $pass)
t = '$THREADS'.split()
random.shuffle(t)
print(' '.join(t))")
    log "[$tag] pass $pass order: $order"
    pos=0
    for th in $order; do
      pos=$((pos + 1))
      "$BIN/llama-bench.exe" -m "$f" -p 128 -n 128 -t "$th" -r "$REPS" -o json \
          > "$WORK/rz_${tag}_${pass}_${th}.json" 2>>"$WORK/rand7b.log"
      python - "$WORK/rz_${tag}_${pass}_${th}.json" "$tag" "$params" "$pass" \
               "$pos" "$bytes" "$sha" >> "$OUT" <<'PY'
import json, sys, statistics as st, time
f, tag, params, p, pos, byts, sha = sys.argv[1:8]
try: rows = json.load(open(f))
except Exception as e:
    print(json.dumps({"tag": tag, "pass": int(p), "error": str(e)})); raise SystemExit
for r in rows:
    s = r.get("samples_ts") or []
    print(json.dumps({
        "tag": tag, "params_B": float(params), "arm": "randomized",
        "pass": int(p), "position_in_pass": int(pos), "threads": r["n_threads"],
        "phase": "decode" if r.get("n_gen", 0) > 0 else "prefill",
        "bytes": int(byts), "sha256": sha, "build": r.get("build_commit"),
        "reps": len(s), "read_at": time.strftime("%H:%M:%S"),
        "tok_s_mean": r["avg_ts"], "tok_s_stddev": r.get("stddev_ts"),
        "tok_s_median": st.median(s) if s else None,
        "samples_ts": s}))
PY
      sleep "$COOL"
    done
    log "[$tag] pass $pass done"
  done

  rm -rf "$f" "$MOD/rz7b"
  log "[$tag] DONE"
done <<< "$TARGETS"

log "=== RANDOMIZED 7B SWEEP DONE ==="
