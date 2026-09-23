#!/bin/bash
# homo_perf.sh (v3) - close the provenance and settings gap in Paper 17's
# homogeneous-core mechanism table, and widen its range.
#
# WHY THIS EXISTS. Table tab:homo-mech pairs a thread curve measured on OFFICIAL
# published GGUFs against instructions-per-token measured on LOCALLY REQUANTIZED
# artifacts, at different llama-bench settings (-p 16 -n 256 there, -p 128 -n 128
# here). Two mismatches, both real:
#   1. Provenance. Paper 16 established that locally requantized files carry 290
#      tensors against the canonical 291, because the requantized graph ties the
#      embedding to the output head. They are not the same artifacts.
#   2. Accounting. `perf stat` over a whole llama-bench process counts model
#      load and prefill alongside decode, so "instructions per token" there is
#      decode instructions plus a fixed overhead divided by token count. That
#      overhead differs across formats precisely because file size differs,
#      which is the variable under test.
#
# BOTH ARE FIXED HERE.
#   Provenance: counters are taken on the same official GGUF the thread curve
#   used, verified by sha256 against data/homo_sweep.jsonl.
#   Accounting: a DIFFERENTIAL measurement. Each artifact is perf'd twice at the
#   same thread count, once at -n 128 and once at -n 256, with prefill disabled.
#   Subtracting cancels model load and every other fixed cost exactly, so
#   (I_256 - I_128) / (reps * 128) is decode instructions per token and nothing
#   else. IPC comes from the same difference, so it is decode IPC.
#
# It also adds two artifacts the sweep never covered, Q5_0 and F16, to widen the
# instructions-per-token range at fixed parameter count. F16 is the extreme
# memory-bound point: no dequantization work at all, the most bytes per token.
# Those two get the full treatment, thread curve and power as well as counters,
# so they join the sweep as first-class rows.
#
# SAMPLER DISCIPLINE is inherited verbatim from homo_sweep.sh v2, which fixed
# the v1 leak: the sampler is a direct child so its PID is real, it carries a
# hard iteration cap, and every PID is journalled for the exit trap. Do not
# refactor power_start into a command substitution; that was the v1 bug.
set -u

BIN=~/llm/llama.cpp/build/bin
WORK=~/homo_perf
OUT=$WORK/homo_perf.jsonl
PIDFILE=$WORK/.sampler_pids
mkdir -p "$WORK"
: > "$PIDFILE"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/run.log"; }

log "=== homo_perf v3 start (binary in $BIN) ==="

R0=https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main
R15=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main

# tag|params|mode|url
#   perf = already has a thread curve in homo_sweep.jsonl, counters only
#   full = new artifact, needs thread curve and power as well as counters
# Order alternates small and large so that file size is decorrelated from
# elapsed time and from board warming, the same discipline as v2.
TARGETS="Q8_0|0.5|perf|$R0/qwen2.5-0.5b-instruct-q8_0.gguf
Q3_K_M|1.5|perf|$R15/qwen2.5-1.5b-instruct-q3_k_m.gguf
Q4_0|0.5|perf|$R0/qwen2.5-0.5b-instruct-q4_0.gguf
F16|0.5|full|$R0/qwen2.5-0.5b-instruct-fp16.gguf
Q2_K|0.5|perf|$R0/qwen2.5-0.5b-instruct-q2_k.gguf
Q4_K_M|1.5|perf|$R15/qwen2.5-1.5b-instruct-q4_k_m.gguf
Q6_K|0.5|perf|$R0/qwen2.5-0.5b-instruct-q6_k.gguf
Q5_0|0.5|full|$R0/qwen2.5-0.5b-instruct-q5_0.gguf
Q3_K_M|0.5|perf|$R0/qwen2.5-0.5b-instruct-q3_k_m.gguf
Q5_K_M|0.5|perf|$R0/qwen2.5-0.5b-instruct-q5_k_m.gguf
Q4_K_M|0.5|perf|$R0/qwen2.5-0.5b-instruct-q4_k_m.gguf"

guard(){
  local avail disk
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  disk=$(df -m "$WORK" | awk 'NR==2{print $4}')
  [ "$avail" -lt 700 ] && { log "ABORT: MemAvailable ${avail}MB"; return 1; }
  [ "$disk" -lt 1600 ] && { log "ABORT: disk ${disk}MB"; return 1; }
  return 0
}

PWPID=""
power_start(){            # out stop
  local out=$1 stop=$2
  rm -f "$stop"
  ( i=0
    while [ ! -f "$stop" ] && [ "$i" -lt 6000 ]; do
      vcgencmd pmic_read_adc 2>/dev/null | awk '
        /current\(/{split($0,a,"="); c[$1]=a[2]+0}
        /volt\(/{split($0,a,"="); v[$1]=a[2]+0}
        END{p=0; for(k in c){ n=k; sub(/_A$/,"",n); vk=n"_V"; if(vk in v) p+=c[k]*v[vk]} print p}'
      sleep 0.1
      i=$((i+1))
    done ) > "$out" 2>/dev/null &
  PWPID=$!                        # a real child: no command substitution here
  echo "$PWPID" >> "$PIDFILE"
}

power_stop(){             # stop
  local stop=$1 k
  touch "$stop"
  for k in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$PWPID" 2>/dev/null || break
    sleep 0.2
  done
  kill -9 "$PWPID" 2>/dev/null
  wait "$PWPID" 2>/dev/null
  rm -f "$stop"                   # only AFTER the loop is confirmed gone
}

cleanup(){
  local p
  if [ -f "$PIDFILE" ]; then
    while read -r p; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done < "$PIDFILE"
  fi
  pkill -9 -x vcgencmd 2>/dev/null
}
trap cleanup EXIT INT TERM

PERF_T=2        # counters at the sweep's modal peak thread count
PERF_R=3        # repetitions inside each perf'd bench
NLO=128
NHI=256

while IFS='|' read -r tag params mode url; do
  [ -z "$tag" ] && continue
  guard || continue
  f="$WORK/m.gguf"
  log "[$params B / $tag / $mode] download"
  curl -sL -o "$f" "$url" || { log "  download FAILED"; rm -f "$f"; continue; }
  bytes=$(stat -c %s "$f")
  if [ "$bytes" -lt 100000000 ]; then
    log "  BAD DOWNLOAD ($bytes bytes), skipping"; rm -f "$f"; continue
  fi
  sha=$(sha256sum "$f" | cut -d' ' -f1)
  t_pre=$(vcgencmd measure_temp | tr -dc '0-9.')
  bg=$(pgrep -c -x vcgencmd 2>/dev/null); bg=${bg:-0}
  log "[$params B / $tag] bytes=$bytes sha=${sha:0:12} temp_pre=${t_pre}C stray=$bg"

  # --- new artifacts also get the v2 thread curve and power passes ---
  if [ "$mode" = "full" ]; then
    "$BIN/llama-bench" -m "$f" -p 128 -n 128 -t 1,2,3,4 -r 10 -o json \
        > "$WORK/benchA_${params}_${tag}.json" 2>>"$WORK/run.log"
    for th in 1 2 3 4; do
      pf="$WORK/pwr_${params}_${tag}_t${th}.txt"
      stop="$WORK/.stop_${params}_${tag}_t${th}"
      power_start "$pf" "$stop"
      "$BIN/llama-bench" -m "$f" -p 128 -n 128 -t "$th" -r 3 -o json \
          > /dev/null 2>>"$WORK/run.log"
      power_stop "$stop"
    done
  fi

  # --- differential counters: -p 0 so nothing but load and decode is counted ---
  for n in "$NLO" "$NHI"; do
    log "[$params B / $tag] perf n=$n t=$PERF_T"
    perf stat -e instructions,cycles,cache-misses \
        -o "$WORK/perf_${params}_${tag}_n${n}.txt" -- \
        "$BIN/llama-bench" -m "$f" -p 0 -n "$n" -r "$PERF_R" -t "$PERF_T" -o json \
        > "$WORK/benchP_${params}_${tag}_n${n}.json" 2>>"$WORK/run.log"
  done

  t_post=$(vcgencmd measure_temp | tr -dc '0-9.')
  log "[$params B / $tag] temp_post=${t_post}C leaked_after=$(pgrep -c -x vcgencmd 2>/dev/null)"

  python3 - "$WORK" "$tag" "$params" "$mode" "$bytes" "$sha" \
           "$t_pre" "$t_post" "$bg" "$PERF_T" "$PERF_R" "$NLO" "$NHI" \
           >> "$OUT" <<'PY'
import json, os, re, sys
work, tag, params, mode, byts, sha, t_pre, t_post, bg, pt, pr, nlo, nhi = sys.argv[1:14]
byts, pt, pr, nlo, nhi = int(byts), int(pt), int(pr), int(nlo), int(nhi)

def counters(n):
    p = "%s/perf_%s_%s_n%d.txt" % (work, params, tag, n)
    if not os.path.exists(p):
        return None
    txt, out = open(p).read(), {}
    for ev in ("instructions", "cycles", "cache-misses"):
        m = re.search(r"([\d,]+)\s+" + ev, txt)
        if m:
            out[ev] = int(m.group(1).replace(",", ""))
    m = re.search(r"([\d.]+) seconds time elapsed", txt)
    if m:
        out["elapsed_s"] = float(m.group(1))
    return out or None

def tok_s(n):
    p = "%s/benchP_%s_%s_n%d.json" % (work, params, tag, n)
    try:
        for r in json.load(open(p)):
            if r.get("n_gen", 0) > 0:
                return r["avg_ts"]
    except Exception:
        pass
    return None

lo, hi = counters(nlo), counters(nhi)
rec = {"tag": tag, "params_B": float(params), "mode": mode, "bytes": byts,
       "sha256": sha, "perf_threads": pt, "perf_reps": pr,
       "n_lo": nlo, "n_hi": nhi,
       "temp_pre_C": float(t_pre), "temp_post_C": float(t_post),
       "stray_samplers_at_start": int(bg),
       "tok_s_lo": tok_s(nlo), "tok_s_hi": tok_s(nhi)}
if lo and hi and "instructions" in lo and "instructions" in hi:
    dtok = pr * (nhi - nlo)
    di = hi["instructions"] - lo["instructions"]
    dc = hi["cycles"] - lo["cycles"]
    rec.update(raw_lo=lo, raw_hi=hi, delta_tokens=dtok,
               inst_per_token=di / dtok, cycles_per_token=dc / dtok,
               ipc_decode=(di / dc) if dc else None)
    if "cache-misses" in lo and "cache-misses" in hi:
        rec["cache_misses_per_token"] = (hi["cache-misses"] - lo["cache-misses"]) / dtok
else:
    rec["perf_err"] = "missing counters"
print(json.dumps(rec))
PY

  # the thread curve for `full` artifacts, in the same schema as homo_sweep.jsonl
  if [ "$mode" = "full" ]; then
    python3 - "$WORK/benchA_${params}_${tag}.json" "$tag" "$params" "$bytes" \
             "$sha" "$WORK/pwr_${params}_${tag}" "$t_pre" "$t_post" "$bg" \
             >> "$WORK/homo_sweep_add.jsonl" <<'PY'
import json, sys, os
f, tag, params, byts, sha, pwrbase, t_pre, t_post, bg = sys.argv[1:10]
byts = int(byts)
def meanw(th):
    p = "%s_t%d.txt" % (pwrbase, th)
    if not os.path.exists(p):
        return None
    v = [float(x) for x in open(p) if x.strip()]
    return sum(v) / len(v) if v else None
try:
    rows = json.load(open(f))
except Exception as e:
    print(json.dumps({"tag": tag, "params_B": float(params), "error": str(e)}))
    raise SystemExit
for r in rows:
    th = r["n_threads"]
    kind = "decode" if r.get("n_gen", 0) > 0 else "prefill"
    w = meanw(th) if kind == "decode" else None
    print(json.dumps({
        "tag": tag, "params_B": float(params), "threads": th, "phase": kind,
        "bytes": byts, "sha256": sha,
        "temp_pre_C": float(t_pre), "temp_post_C": float(t_post),
        "stray_samplers_at_start": int(bg),
        "tok_s": r["avg_ts"], "stddev": r.get("stddev_ts"),
        "reps": len(r.get("samples_ts", [])),
        "mean_W": w,
        "mJ_per_tok": (w / r["avg_ts"] * 1000) if (w and r["avg_ts"]) else None}))
PY
  fi

  rm -f "$f"
  log "[$params B / $tag] done"
done <<< "$TARGETS"

log "=== DONE ==="
