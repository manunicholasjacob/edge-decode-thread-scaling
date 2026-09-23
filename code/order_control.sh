#!/bin/bash
# order_control.sh - is the decode cliff a property of thread count, or of
# WHEN in the session each thread count was measured?
#
# WHY THIS NOW OUTRANKS EVERYTHING ELSE. Two runs of the same file, same binary,
# same machine, ten repetitions each, disagree like this:
#
#   t        1     4     8    12    16    20
#   run 1  20.9  75.0  90.7 101.2 106.8  90.6     (whole sweep in 6.5 min, cold)
#   run 2  21.3  81.2  84.2  90.1  66.7  34.8     (same sweep, later, warm)
#
# They agree at one thread and diverge monotonically with position in the sweep,
# reaching 61% at twenty threads. Thread count is confounded with elapsed time
# because the sweep always walks thread counts in ascending order, so the
# highest counts are always measured last, on the hottest machine. A sequential
# sweep on a thermally limited laptop cannot tell "more threads is slower" from
# "later is slower".
#
# This is not a novel worry. It is the subject of Paper 2 in this same
# portfolio, which shows on a consumer GPU that a blocked design confounds
# identity with thermal drift, that randomising converts the bias into variance,
# and that the residual tracks achieved clock rather than temperature. Paper 17
# has been committing the error Paper 2 documents.
#
# THE TEST. Measure the same thread counts twice in one session, ascending then
# descending, and bracket both with a fixed reference cell.
#
#   * If the cliff sits at high thread counts in BOTH passes, it is a property
#     of thread count and the paper's effect is real.
#   * If it follows the END of the pass, appearing at high counts ascending and
#     at LOW counts descending, it is drift and the headline is an artifact of
#     measurement order.
#   * The three reference cells, identical work at t=4 measured at the start,
#     middle and end, quantify the drift directly regardless of which way the
#     sweep goes. If the reference decays, every sequential number in this paper
#     is suspect.
#
# Nothing else should run on this machine while this runs.
set -u

BIN=/c/llmpc/bin
MOD=/c/llmpc/models
WORK=/c/llmpc/sweep
M=$MOD/oc.gguf
OUT=$WORK/order_control.jsonl
mkdir -p "$MOD" "$WORK"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/order.log"; }

URL=https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
REPS=10
ASC=1,2,3,4,5,6,7,8,9,10,12,14,16,18,20
DESC=20,18,16,14,12,10,9,8,7,6,5,4,3,2,1
COOL=180          # seconds idle between passes, to let the package settle

fetch(){ local k; for k in $(seq 1 15); do
  curl -sSL --retry 3 --retry-delay 10 --connect-timeout 30 -o "$M" "$URL" 2>/dev/null \
    && [ -s "$M" ] && return 0; log "  fetch $k failed"; sleep 30; done; return 1; }

[ -s "$M" ] || fetch || { log "DOWNLOAD FAILED"; exit 1; }
BYTES=$(stat -c %s "$M"); SHA=$(sha256sum "$M" | cut -d' ' -f1)
: > "$OUT"
log "=== order control start: bytes=$BYTES sha=${SHA:0:12} ==="

emit(){                      # jsonfile arm
  python - "$1" "$2" "$BYTES" "$SHA" >> "$OUT" <<'PY'
import json, sys, statistics as st, time
f, arm, byts, sha = sys.argv[1:5]
try: rows = json.load(open(f))
except Exception as e:
    print(json.dumps({"arm": arm, "error": str(e)})); raise SystemExit
for i, r in enumerate(rows):
    s = r.get("samples_ts") or []
    print(json.dumps({
        "tag": "qwen0.5b", "arm": arm, "cell_index": i,
        "threads": r["n_threads"],
        "phase": "decode" if r.get("n_gen", 0) > 0 else "prefill",
        "bytes": int(byts), "sha256": sha, "build": r.get("build_commit"),
        "reps": len(s), "read_at": time.strftime("%H:%M:%S"),
        "tok_s_mean": r["avg_ts"], "tok_s_stddev": r.get("stddev_ts"),
        "tok_s_median": st.median(s) if s else None,
        "tok_s_min": min(s) if s else None, "tok_s_max": max(s) if s else None,
        "samples_ts": s}))
PY
}

ref(){                       # label
  log "reference cell ($1)"
  "$BIN/llama-bench.exe" -m "$M" -p 128 -n 128 -t 4 -r "$REPS" -o json \
      > "$WORK/oc_ref_$1.json" 2>>"$WORK/order.log"
  emit "$WORK/oc_ref_$1.json" "ref_$1"
}

ref start
log "pass 1: ASCENDING $ASC"
"$BIN/llama-bench.exe" -m "$M" -p 128 -n 128 -t "$ASC" -r "$REPS" -o json \
    > "$WORK/oc_asc.json" 2>>"$WORK/order.log"
emit "$WORK/oc_asc.json" "ascending"

ref middle
log "cooling ${COOL}s"
sleep "$COOL"

log "pass 2: DESCENDING $DESC"
"$BIN/llama-bench.exe" -m "$M" -p 128 -n 128 -t "$DESC" -r "$REPS" -o json \
    > "$WORK/oc_desc.json" 2>>"$WORK/order.log"
emit "$WORK/oc_desc.json" "descending"

ref end
rm -f "$M"
log "=== ORDER CONTROL DONE ==="

python - "$OUT" <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
dec = [r for r in rows if r.get("phase") == "decode" and "error" not in r]
by = collections.defaultdict(dict)
for r in dec:
    by[r["arm"]][r["threads"]] = r["tok_s_median"]

refs = [(a, list(d.values())[0]) for a, d in by.items() if a.startswith("ref_")]
print("\nreference cell (t=4, identical work):")
for a, v in refs:
    print("   %-10s %.2f tok/s" % (a.replace("ref_", ""), v))
if len(refs) > 1:
    lo = min(v for _, v in refs); hi = max(v for _, v in refs)
    print("   drift across the session: %.1f%%" % (100 * (1 - lo / hi)))
    if 100 * (1 - lo / hi) > 10:
        print("   *** the machine drifts; sequential sweeps are confounded ***")

print("\n%5s %12s %12s %8s" % ("t", "ascending", "descending", "diff%"))
a, d = by.get("ascending", {}), by.get("descending", {})
for t in sorted(set(a) & set(d)):
    print("%5d %12.2f %12.2f %+7.1f%%" % (t, a[t], d[t], 100 * (d[t] / a[t] - 1)))
for nm, dd in (("ascending", a), ("descending", d)):
    if not dd:
        continue
    pk = max(dd, key=dd.get)
    print("%-11s peak %.1f at t=%d ; t=20 %.1f ; fall from peak %.1f%%"
          % (nm, dd[pk], pk, dd[20], 100 * (1 - dd[20] / dd[pk])))
print("\nIf the fall follows the end of the pass rather than the thread count,")
print("the cliff is measurement order, not hybrid cores.")
PY
