#!/bin/bash
# headroom_test.sh - is the tail collapse caused by running out of logical
# processors, or by something else?
#
# THE HYPOTHESIS THE PAPER CURRENTLY STATES WITHOUT TESTING. Under the
# randomised protocol, decode throughput peaks at sixteen threads and past that
# the MEDIAN barely moves while the LOWER TAIL collapses: at eighteen threads
# the upper quartile is 95% of the peak's and the minimum is 36% of it, with the
# coefficient of variation going from 2.7% at sixteen threads to 29.4% at
# eighteen and 31.0% at twenty. The paper suggests headroom: this machine has
# twenty logical processors, at sixteen threads four stay free for the operating
# system, at twenty none do, and because a decode step joins at a barrier one
# descheduled thread stalls the whole step. That produces a usually-fine and
# occasionally-terrible distribution, which is what we see.
#
# It is a story that fits. It has not been tested, and the paper says so.
#
# THE TEST. Hold the thread count at sixteen and remove the headroom by hand.
#
#   A  t=16 on logical processors 0-15, nothing else running        baseline
#   B  t=16 on logical processors 0-15, four spinners pinned to     the test
#      processors 16-19, so the four free processors are occupied
#   C  t=20 on all twenty, nothing else running                     the tail
#   D  A again                                                      drift check
#
# If B looks like C, the mechanism is contention for logical processors and the
# deployment advice is to leave headroom. If B still looks like A, then
# occupying the spare processors is not what hurts, the headroom account is
# wrong, and something about the runtime's own behaviour at full occupancy is
# responsible instead. Either answer is worth having; only one of them is the
# answer the paper currently guesses.
#
# ARM ORDER IS PERMUTED PER PASS, for the reason this whole line of work exists:
# a fixed order would confound arm with elapsed time.
#
# The masks are the ones Paper 17's mask validation established as binding:
# confining eight threads to one logical processor gives 0.16 of eight free
# threads on decode, so --cpu-mask is doing what it claims on this build.
set -u

BIN=/c/llmpc/bin
MOD=/c/llmpc/models
WORK=/c/llmpc/headroom
OUT=$WORK/headroom_test.jsonl
M=$MOD/hr.gguf
mkdir -p "$MOD" "$WORK"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/run.log"; }

URL=https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
PASSES=4
REPS=10
COOL=25
MASK16=0xffff        # logical processors 0-15
MASK20=0xfffff       # all twenty
SPIN_CPUS="10000 20000 40000 80000"   # one spinner per free processor, 16..19

fetch(){ local k; for k in $(seq 1 15); do
  curl -sSL --retry 3 --retry-delay 10 --connect-timeout 30 -o "$M" "$URL" 2>/dev/null \
    && [ -s "$M" ] && return 0; log "  fetch $k failed"; sleep 20; done; return 1; }

SPIN_STARTED=0
# cmd //c start /affinity blocks in this shell, so the spinners are launched and
# pinned from PowerShell, which also journals their PIDs to a file. The spinner
# itself takes a deadline and exits on its own: Paper 17's first Pi campaign
# leaked 36 samplers that ran for hours, and a worker that cannot outlive its
# own timeout is the cheap way not to repeat that.
spin_up(){
  powershell -NoProfile -ExecutionPolicy Bypass -File C:/llmpc/spin_up.ps1 -Seconds 900       >> "$WORK/run.log" 2>&1
  SPIN_STARTED=1
  sleep 2
  log "  spinners up ($(ps -W 2>/dev/null | grep -ci python) python procs)"
}
spin_down(){
  [ "$SPIN_STARTED" -eq 0 ] && return
  powershell -NoProfile -ExecutionPolicy Bypass -File C:/llmpc/spin_down.ps1       >> "$WORK/run.log" 2>&1
  SPIN_STARTED=0
  sleep 2
  log "  spinners down"
}
trap 'spin_down; rm -f "$M"' EXIT INT TERM

[ -s "$M" ] || fetch || { log "DOWNLOAD FAILED"; exit 1; }
BYTES=$(stat -c %s "$M"); SHA=$(sha256sum "$M" | cut -d' ' -f1)
: > "$OUT"
log "=== headroom test: $PASSES passes, r=$REPS ==="

run_arm(){                    # arm threads mask loaded pass pos
  local arm=$1 th=$2 mask=$3 loaded=$4 pass=$5 pos=$6
  [ "$loaded" = "1" ] && spin_up
  "$BIN/llama-bench.exe" -m "$M" -p 128 -n 128 -t "$th" -r "$REPS" \
      -C "$mask" --cpu-strict 1 -o json \
      > "$WORK/hr_${arm}_${pass}.json" 2>>"$WORK/run.log"
  [ "$loaded" = "1" ] && spin_down
  python - "$WORK/hr_${arm}_${pass}.json" "$arm" "$th" "$mask" "$loaded" \
           "$pass" "$pos" "$BYTES" "$SHA" >> "$OUT" <<'PY'
import json, sys, statistics as st, time
f, arm, th, mask, loaded, p, pos, byts, sha = sys.argv[1:10]
try: rows = json.load(open(f))
except Exception as e:
    print(json.dumps({"arm": arm, "pass": int(p), "error": str(e)})); raise SystemExit
for r in rows:
    s = r.get("samples_ts") or []
    print(json.dumps({
        "arm": arm, "threads": r["n_threads"], "cpu_mask": mask,
        "background_load": int(loaded), "pass": int(p),
        "position_in_pass": int(pos),
        "phase": "decode" if r.get("n_gen", 0) > 0 else "prefill",
        "bytes": int(byts), "sha256": sha, "build": r.get("build_commit"),
        "reps": len(s), "read_at": time.strftime("%H:%M:%S"),
        "tok_s_mean": r["avg_ts"], "tok_s_stddev": r.get("stddev_ts"),
        "tok_s_median": st.median(s) if s else None,
        "tok_s_min": min(s) if s else None, "samples_ts": s}))
PY
  log "  $arm pass=$pass pos=$pos done"
}

for pass in $(seq 1 "$PASSES"); do
  order=$(python -c "
import random
random.seed(400 + $pass)
a = ['A_16_free', 'B_16_loaded', 'C_20_free', 'D_16_repeat']
random.shuffle(a)
print(' '.join(a))")
  log "pass $pass order: $order"
  pos=0
  for arm in $order; do
    pos=$((pos + 1))
    case "$arm" in
      A_16_free)   run_arm "$arm" 16 "$MASK16" 0 "$pass" "$pos" ;;
      B_16_loaded) run_arm "$arm" 16 "$MASK16" 1 "$pass" "$pos" ;;
      C_20_free)   run_arm "$arm" 20 "$MASK20" 0 "$pass" "$pos" ;;
      D_16_repeat) run_arm "$arm" 16 "$MASK16" 0 "$pass" "$pos" ;;
    esac
    sleep "$COOL"
  done
done

rm -f "$M"
log "=== HEADROOM TEST DONE ==="

python - "$OUT" <<'PY'
import collections, json, statistics as st, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
dec = [r for r in rows if r.get("phase") == "decode" and "error" not in r]
by = collections.defaultdict(list)
for r in dec:
    by[r["arm"]].extend(r["samples_ts"])
print("\n%-14s %8s %8s %8s %7s %s" % ("arm", "median", "p25", "min", "CV%", "n"))
def pct(v, p):
    s = sorted(v); k = (len(s)-1)*p/100.0; lo = int(k); hi = min(lo+1, len(s)-1)
    return s[lo] + (s[hi]-s[lo])*(k-lo)
for a in sorted(by):
    v = by[a]
    print("%-14s %8.1f %8.1f %8.1f %7.1f %d"
          % (a, st.median(v), pct(v, 25), min(v), 100*st.pstdev(v)/st.mean(v), len(v)))
A, B, C = by.get("A_16_free"), by.get("B_16_loaded"), by.get("C_20_free")
if A and B and C:
    cv = lambda v: 100*st.pstdev(v)/st.mean(v)
    print("\nCV: A(16 free) %.1f%%  B(16 loaded) %.1f%%  C(20 free) %.1f%%"
          % (cv(A), cv(B), cv(C)))
    if cv(B) > (cv(A) + cv(C)) / 2:
        print("-> Occupying the spare processors REPRODUCES the tail.")
        print("   Headroom is the mechanism; leave processors free.")
    else:
        print("-> Occupying the spare processors does NOT reproduce the tail.")
        print("   The headroom account is wrong and the paper must say so.")
PY
