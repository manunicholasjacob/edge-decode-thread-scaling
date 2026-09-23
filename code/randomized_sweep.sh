#!/bin/bash
# randomized_sweep.sh - measure the thread curve without confounding thread
# count with elapsed time.
#
# THE PROBLEM THIS EXISTS FOR. Every laptop sweep in this paper so far walks
# thread counts in ascending order inside a single llama-bench invocation, so
# the highest thread counts are always measured last, on the hottest machine.
# Two runs of the identical file then disagree by 61% at twenty threads while
# agreeing at one, with the gap growing monotonically along the sweep. On a
# thermally limited laptop that design cannot separate "more threads is slower"
# from "later is slower", and the paper's headline is a difference between the
# beginning and the end of a sweep.
#
# Paper 2 in this portfolio documents exactly this failure on a consumer GPU,
# shows that randomising converts the bias into variance, and prices the
# thermal-gating remedy at a 30% yield cost. This applies its lesson.
#
# THE DESIGN.
#   * PASSES independent passes over the same thread counts.
#   * Each pass visits them in a DIFFERENT random permutation, so a given thread
#     count is measured early in one pass and late in another.
#   * A fixed cooldown between cells, and each cell run as its own llama-bench
#     invocation rather than as one call with a -t list, because a list is
#     exactly the contiguous ordering being removed.
#   * Every cell records which pass it belonged to and its position within that
#     pass, so drift can be measured after the fact instead of assumed absent.
#
# What comes out is, per thread count, PASSES x REPS samples spread across
# different session positions. The median over those is unbiased with respect to
# order, and regressing value on position quantifies whatever drift remains.
# That is a claim the ascending sweep cannot make at any repetition count,
# because repeating a biased measurement does not remove the bias.
set -u

BIN=/c/llmpc/bin
MOD=/c/llmpc/models
WORK=/c/llmpc/sweep
M=$MOD/rs.gguf
OUT=$WORK/randomized_sweep.jsonl
mkdir -p "$MOD" "$WORK"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/rand.log"; }

URL=https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
THREADS="1 2 3 4 5 6 7 8 9 10 12 14 16 18 20"
PASSES=5
REPS=3
COOL=20

fetch(){ local k; for k in $(seq 1 20); do
  curl -sSL --retry 3 --retry-delay 10 --connect-timeout 30 -o "$M" "$URL" 2>/dev/null \
    && [ -s "$M" ] && return 0; log "  fetch $k failed"; sleep 30; done; return 1; }

[ -s "$M" ] || fetch || { log "DOWNLOAD FAILED"; exit 1; }
BYTES=$(stat -c %s "$M"); SHA=$(sha256sum "$M" | cut -d' ' -f1)
: > "$OUT"
log "=== randomized sweep: $PASSES passes, r=$REPS, ${COOL}s cooldown ==="

for pass in $(seq 1 "$PASSES"); do
  # A fresh permutation per pass. Seeded by the pass number so the whole
  # schedule is reproducible from the script alone.
  order=$(python -c "
import random
random.seed(1000 + $pass)
t = '$THREADS'.split()
random.shuffle(t)
print(' '.join(t))")
  log "pass $pass order: $order"
  pos=0
  for th in $order; do
    pos=$((pos + 1))
    "$BIN/llama-bench.exe" -m "$M" -p 128 -n 128 -t "$th" -r "$REPS" -o json \
        > "$WORK/rs_${pass}_${th}.json" 2>>"$WORK/rand.log"
    python - "$WORK/rs_${pass}_${th}.json" "$pass" "$pos" "$BYTES" "$SHA" >> "$OUT" <<'PY'
import json, sys, statistics as st, time
f, p, pos, byts, sha = sys.argv[1:6]
try: rows = json.load(open(f))
except Exception as e:
    print(json.dumps({"pass": int(p), "error": str(e)})); raise SystemExit
for r in rows:
    s = r.get("samples_ts") or []
    print(json.dumps({
        "tag": "qwen0.5b", "arm": "randomized", "pass": int(p),
        "position_in_pass": int(pos), "threads": r["n_threads"],
        "phase": "decode" if r.get("n_gen", 0) > 0 else "prefill",
        "bytes": int(byts), "sha256": sha, "build": r.get("build_commit"),
        "reps": len(s), "read_at": time.strftime("%H:%M:%S"),
        "tok_s_mean": r["avg_ts"], "tok_s_stddev": r.get("stddev_ts"),
        "tok_s_median": st.median(s) if s else None,
        "samples_ts": s}))
PY
    sleep "$COOL"
  done
  log "pass $pass done"
done

rm -f "$M"
log "=== RANDOMIZED SWEEP DONE ==="

python - "$OUT" <<'PY'
import collections, json, statistics as st, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
dec = [r for r in rows if r.get("phase") == "decode" and "error" not in r]
by = collections.defaultdict(list)
pos = collections.defaultdict(list)
for r in dec:
    by[r["threads"]].extend(r["samples_ts"])
    pos[r["position_in_pass"]].extend(r["samples_ts"])

print("\n%5s %9s %9s %9s %6s" % ("t", "median", "min", "max", "n"))
for t in sorted(by):
    v = by[t]
    print("%5d %9.2f %9.2f %9.2f %6d" % (t, st.median(v), min(v), max(v), len(v)))

if by:
    pk = max(by, key=lambda t: st.median(by[t]))
    hi = max(by)
    print("\npeak %.1f tok/s at t=%d ; t=%d %.1f ; fall %.1f%%"
          % (st.median(by[pk]), pk, hi, st.median(by[hi]),
             100 * (1 - st.median(by[hi]) / st.median(by[pk]))))

# Drift check: does position within a pass predict throughput, with thread
# count averaged out by the randomisation?
print("\nthroughput by POSITION in pass (thread count randomised away):")
for p in sorted(pos):
    print("   position %2d  median %.2f  n=%d" % (p, st.median(pos[p]), len(pos[p])))
first = st.median(pos[min(pos)]) if pos else 0
last = st.median(pos[max(pos)]) if pos else 0
if first:
    print("   first-to-last position change: %+.1f%%" % (100 * (last / first - 1)))
    print("   (a large negative number here is drift, not thread scaling)")
PY
