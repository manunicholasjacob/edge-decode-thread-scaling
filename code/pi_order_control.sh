#!/bin/bash
# pi_order_control.sh - does the Pi's thread curve have the same order confound
# the laptop's does?
#
# WHY ASK. The laptop sweeps are confounded: thread count always ascends inside
# one llama-bench invocation, so the highest counts are measured last on the
# hottest machine, and two runs of the identical file disagree by 61% at twenty
# threads. The Pi campaigns were careful about a different confound, alternating
# small and large ARTIFACTS so that file size is decorrelated from elapsed time,
# and that discipline is why the mechanism study is trustworthy. But within each
# artifact they run `-t 1,2,3,4` in one call, ascending, exactly like the laptop.
#
# Every "decline" this paper reports for homogeneous cores is peak-to-four
# threads, and four threads is always the LAST cell measured for that artifact.
# If the Pi drifts the way the laptop does, that decline is partly drift too, and
# the sixteen-artifact mechanism result inherits the problem.
#
# The Pi has two things the laptop lacks: a real temperature reading, and a
# fixed clock ceiling that it reaches under sustained load. So this can measure
# the confound directly rather than infer it.
#
# THE TEST, on one mid-sized artifact:
#   ref (t=2) -> ascending 1,2,3,4 -> ref -> cooldown -> descending 4,3,2,1 -> ref
# recording SoC temperature and throttle flags around every cell.
#
#   * If the four-thread value is the same in both directions, the Pi's decline
#     is a property of thread count and the mechanism study stands as written.
#   * If four threads looks worse when it comes last, the decline is inflated by
#     drift and every Pi number needs the randomised treatment too.
#
# vcgencmd get_throttled is the thing to watch: bit 0 is under-voltage now, bit
# 1 is arm frequency capped now, bit 2 is currently throttled, bit 3 is soft
# temperature limit active. A non-zero low nibble during the run means the board
# was actively limited and the numbers describe the limiter, not the workload.
set -u

BIN=~/llm/llama.cpp/build/bin
WORK=~/pi_order
M=$WORK/m.gguf
OUT=$WORK/pi_order_control.jsonl
mkdir -p "$WORK"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/run.log"; }

URL=https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
REPS=10
COOL=240

fetch(){ local k; for k in $(seq 1 10); do
  curl -sSL --retry 3 --retry-delay 10 -o "$M" "$URL" 2>/dev/null && [ -s "$M" ] && return 0
  log "  fetch $k failed"; sleep 20; done; return 1; }

[ -s "$M" ] || fetch || { log "DOWNLOAD FAILED"; exit 1; }
BYTES=$(stat -c %s "$M"); SHA=$(sha256sum "$M" | cut -d' ' -f1)
: > "$OUT"
log "=== pi order control: bytes=$BYTES sha=${SHA:0:12} ==="

state(){ printf '%s %s' \
  "$(vcgencmd measure_temp | tr -dc '0-9.')" \
  "$(vcgencmd get_throttled | sed 's/.*=//')"; }

cell(){                      # arm threads position
  local arm=$1 th=$2 pos=$3 pre post
  pre=$(state)
  "$BIN/llama-bench" -m "$M" -p 128 -n 128 -t "$th" -r "$REPS" -o json \
      > "$WORK/c_${arm}_${th}.json" 2>>"$WORK/run.log"
  post=$(state)
  python3 - "$WORK/c_${arm}_${th}.json" "$arm" "$th" "$pos" "$BYTES" "$SHA" \
           "$pre" "$post" >> "$OUT" <<'PY'
import json, sys, statistics as st, time
f, arm, th, pos, byts, sha, pre, post = sys.argv[1:9]
pre_t, pre_thr = pre.split()
post_t, post_thr = post.split()
try: rows = json.load(open(f))
except Exception as e:
    print(json.dumps({"arm": arm, "threads": int(th), "error": str(e)})); raise SystemExit
for r in rows:
    s = r.get("samples_ts") or []
    print(json.dumps({
        "tag": "qwen0.5b", "arm": arm, "threads": r["n_threads"],
        "position": int(pos),
        "phase": "decode" if r.get("n_gen", 0) > 0 else "prefill",
        "bytes": int(byts), "sha256": sha, "reps": len(s),
        "temp_pre_C": float(pre_t), "temp_post_C": float(post_t),
        "throttled_pre": pre_thr, "throttled_post": post_thr,
        "read_at": time.strftime("%H:%M:%S"),
        "tok_s_mean": r["avg_ts"], "tok_s_stddev": r.get("stddev_ts"),
        "tok_s_median": st.median(s) if s else None,
        "samples_ts": s}))
PY
  log "  $arm t=$th pos=$pos  pre=[$pre] post=[$post]"
}

cell ref_start 2 0
p=0; for th in 1 2 3 4; do p=$((p+1)); cell ascending "$th" "$p"; done
cell ref_middle 2 0
log "cooling ${COOL}s"
sleep "$COOL"
p=0; for th in 4 3 2 1; do p=$((p+1)); cell descending "$th" "$p"; done
cell ref_end 2 0

rm -f "$M"
log "=== PI ORDER CONTROL DONE ==="

python3 - "$OUT" <<'PY'
import collections, json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
dec = [r for r in rows if r.get("phase") == "decode" and "error" not in r]
by = collections.defaultdict(dict)
temp = {}
for r in dec:
    by[r["arm"]][r["threads"]] = r["tok_s_median"]
    temp[(r["arm"], r["threads"])] = (r["temp_pre_C"], r["temp_post_C"],
                                      r["throttled_post"])
print("\nreference cells (t=2, identical work):")
for a in ("ref_start", "ref_middle", "ref_end"):
    if a in by:
        v = list(by[a].values())[0]
        t = temp.get((a, 2))
        print("   %-11s %.2f tok/s   temp %.1f -> %.1f C  throttled=%s"
              % (a.replace("ref_", ""), v, t[0], t[1], t[2]))
refs = [list(by[a].values())[0] for a in ("ref_start", "ref_middle", "ref_end") if a in by]
if len(refs) > 1:
    print("   drift across session: %.1f%%" % (100 * (1 - min(refs) / max(refs))))

a, d = by.get("ascending", {}), by.get("descending", {})
print("\n%5s %12s %12s %8s" % ("t", "ascending", "descending", "diff%"))
for t in sorted(set(a) & set(d)):
    print("%5d %12.2f %12.2f %+7.1f%%" % (t, a[t], d[t], 100 * (d[t] / a[t] - 1)))
for nm, dd in (("ascending", a), ("descending", d)):
    if dd:
        pk = max(dd, key=dd.get)
        print("%-11s peak %.2f at t=%d ; t=4 %.2f ; decline %.1f%%"
              % (nm, dd[pk], pk, dd[4], 100 * (1 - dd[4] / dd[pk])))
print("\nIf the two declines differ materially, the Pi decline is drift too.")
PY
