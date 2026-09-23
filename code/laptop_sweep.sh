#!/bin/bash
# laptop_sweep.sh (v2) - re-measure Paper 17's laptop evidence to answer the
# CAL referees directly, on the same i7-12700H that produced the original.
#
# WHAT THE REFEREES SAID, and what each phase here does about it.
#
#   "Figure 1 has no data point at 6 threads (the P-core count) or 7 (the first
#   E-core), which is exactly where the claimed cliff must appear."
#     -> PHASE A sweeps every thread count from 1 to 10 and then to 20. The
#        published sweep had five points (1,4,8,14,20) and stepped straight over
#        the boundary the paper is about.
#
#   "There is no thread pinning, so the causal attribution to E-cores is not
#   established."
#     -> PHASE B measures one thread pinned to each of the 20 logical processors
#        in turn, which identifies P and E cores from throughput instead of
#        assuming an enumeration order, and quantifies the per-core penalty.
#     -> PHASE C then holds the thread count fixed and moves only the placement.
#        If throughput at eight threads recovers when the mask excludes the
#        E-cores, the attribution is established rather than inferred.
#
#   "Three repetitions with no standard deviations."
#     -> Ten repetitions everywhere, and every sample is kept, so medians and
#        spreads are computable rather than promised.
#
# It also fixes what auditing the old table found (TABLE_AUDIT_2026-09-17.md):
# several 8-thread cells printed a maximum or a mean where the text claimed a
# median, and the Llama 1.2B row had no released records at all.
#
# PROVENANCE. The original campaign's model files are gone from this machine, so
# these are fresh downloads from the publishers' repositories with sha256 and
# source URL recorded per file. Paper 16 established that two files with the
# same label need not be the same file.
#
# SAFETY. One model resident at a time, deleted after its phases. The 7B model
# ships as two shards and llama.cpp opens it from the first.
set -u

BIN=/c/llmpc/bin
MOD=/c/llmpc/models
WORK=/c/llmpc/sweep
PY=python
mkdir -p "$MOD" "$WORK"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/run.log"; }

SWEEP_T=1,2,3,4,5,6,7,8,9,10,12,14,16,18,20
REPS=10
PROMPT=128
GEN=128
NLOGICAL=20

log "=== laptop_sweep v2 start: threads=$SWEEP_T reps=$REPS ==="

HF=https://huggingface.co
# tag|params|url   (7B is sharded)
TARGETS="qwen0.5b|0.494|$HF/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
llama1.2b|1.236|$HF/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
qwen1.5b|1.544|$HF/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
qwen3b|3.086|$HF/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
qwen7b|7.616|$HF/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

freegb(){ df -m "$MOD" | awk 'NR==2{printf "%.1f", $4/1024}'; }

# Downloads fail transiently (one did on the v1 run, inside the same second it
# started). Retry rather than silently dropping a model from the table.
fetch(){                      # url dest
  local u=$1 d=$2 k
  for k in 1 2 3 4 5; do
    if curl -sSL --retry 3 --retry-delay 5 -o "$d" "$u" 2>>"$WORK/run.log"; then
      [ -s "$d" ] && return 0
    fi
    log "    fetch attempt $k failed, retrying"
    sleep 10
  done
  return 1
}

# One record per llama-bench row, with every sample kept.
emit(){                       # json tag params bytes sha url phaselabel mask
  "$PY" - "$@" <<'PY'
import json, sys, statistics as st
f, tag, params, byts, sha, url, phase_label, mask = sys.argv[1:9]
try:
    rows = json.load(open(f))
except Exception as e:
    print(json.dumps({"tag": tag, "arm": phase_label, "error": str(e)}))
    raise SystemExit
for r in rows:
    s = r.get("samples_ts") or []
    kind = "decode" if r.get("n_gen", 0) > 0 else "prefill"
    print(json.dumps({
        "tag": tag, "params_B": float(params), "arm": phase_label,
        "cpu_mask": mask, "threads": r["n_threads"], "phase": kind,
        "bytes": int(byts), "sha256": sha, "source_url": url,
        "build": r.get("build_commit"), "reps": len(s),
        "tok_s_mean": r["avg_ts"], "tok_s_stddev": r.get("stddev_ts"),
        "tok_s_median": st.median(s) if s else None,
        "tok_s_min": min(s) if s else None,
        "tok_s_max": max(s) if s else None,
        "samples_ts": s}))
PY
}

while IFS='|' read -r tag params url; do
  [ -z "$tag" ] && continue
  f="$MOD/$tag.gguf"
  log "[$tag] download ($(freegb) GB free)"
  fetch "$url" "$f" || { log "  DOWNLOAD FAILED after retries"; rm -f "$f"; continue; }

  if [ "$tag" = "qwen7b" ]; then
    d="$MOD/qwen7b-split"; mkdir -p "$d"
    mv "$f" "$d/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    fetch "$HF/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf" \
          "$d/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf" \
      || { log "  SHARD 2 FAILED"; rm -rf "$d"; continue; }
    f="$d/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    bytes=$(( $(stat -c %s "$d"/*00001*) + $(stat -c %s "$d"/*00002*) ))
  else
    bytes=$(stat -c %s "$f")
  fi
  if [ "$bytes" -lt 100000000 ]; then
    log "  BAD DOWNLOAD ($bytes bytes)"; rm -rf "$f" "$MOD/qwen7b-split"; continue
  fi
  sha=$(sha256sum "$f" | cut -d' ' -f1)
  log "[$tag] bytes=$bytes sha=${sha:0:12}"

  # ---------------- PHASE B: which logical processor is which ----------------
  # Only the smallest model, because this is a property of the cores and one
  # thread on 0.5B resolves it in seconds per core. --cpu-strict 1 makes the
  # mask binding rather than advisory.
  #
  # CLASSIFY ON PREFILL, NOT DECODE. A pilot found single-thread decode
  # essentially equal on a fast and a slow logical processor (23.3 against 24.0
  # tok/s) while compute-bound prefill separated them clearly (34.6 against
  # 25.5). That is the two-term account showing up at n=1: one core of either
  # kind cannot saturate the bus, so decode alone cannot tell the core types
  # apart. It is also a result worth reporting, because it means the E-core
  # penalty in decode is not a slower per-core decode rate.
  #
  # TWO PASSES IN OPPOSITE ORDERS. The same pilot put the two highest scores on
  # the LAST two processors tested, which is what thermal or turbo drift over a
  # sequential scan looks like and is indistinguishable from a real core-type
  # difference within one pass. Scanning 0..19 and then 19..0 separates them: a
  # genuine per-core property is stable between the passes, drift reverses.
  if [ "$tag" = "qwen0.5b" ]; then
    for pass in up down; do
      log "[$tag] PHASE B pass=$pass: per-logical-core identification"
      if [ "$pass" = "up" ]; then order=$(seq 0 $((NLOGICAL-1)))
      else order=$(seq $((NLOGICAL-1)) -1 0); fi
      for i in $order; do
        m=$("$PY" -c "print(hex(1 << $i))")
        "$BIN/llama-bench.exe" -m "$f" -p "$PROMPT" -n "$GEN" -t 1 -r 5 \
            -C "$m" --cpu-strict 1 -o json \
            > "$WORK/coreid_${pass}_$i.json" 2>>"$WORK/run.log"
        emit "$WORK/coreid_${pass}_$i.json" "$tag" "$params" "$bytes" "$sha" "$url" \
             "coreid_$pass" "$m" >> "$WORK/laptop_coreid.jsonl"
      done
    done
    log "[$tag] PHASE B done (both passes)"
  fi

  # ---------------- PHASE A: the dense unpinned sweep -----------------------
  log "[$tag] PHASE A: sweep $SWEEP_T"
  "$BIN/llama-bench.exe" -m "$f" -p "$PROMPT" -n "$GEN" \
      -t "$SWEEP_T" -r "$REPS" -o json \
      > "$WORK/bench_$tag.json" 2>>"$WORK/run.log"
  emit "$WORK/bench_$tag.json" "$tag" "$params" "$bytes" "$sha" "$url" \
       "default" "none" >> "$WORK/laptop_sweep.jsonl"
  log "[$tag] PHASE A done"

  rm -rf "$f" "$MOD/qwen7b-split"
done <<< "$TARGETS"

log "=== DONE phases A and B: $(wc -l < "$WORK/laptop_sweep.jsonl") sweep rows ==="
log "Phase C (placement control) runs separately, once Phase B has named the cores."
