#!/bin/bash
# homo_perf_15b.sh - extend the homogeneous-core study's weakest claim.
#
# WHY. fill_mech.py reports that instructions per token predict the thread
# optimum well (r=+0.75 over eleven artifacts, +0.80 within the nine that share
# the 0.5 B parameter count) but predict the DEPTH of the decline much less well
# once the 1.5 B artifacts join: r=-0.73 within the 0.5 B family against -0.38
# over all eleven. The paper says so and declines to model the depth.
#
# That scoping rests on exactly TWO 1.5 B artifacts. Two points cannot
# distinguish "the relationship differs at 1.5 B" from "two points scattered".
# This adds four more so the 1.5 B family has six members and can be tested
# within itself, the same way the 0.5 B family already is. Either the
# within-family relationship reappears at 1.5 B, which strengthens the mechanism
# and narrows the caveat to a between-family effect, or it does not, which makes
# the caveat a finding rather than an apology.
#
# Q8_0 is deliberately excluded at this size: the file is about 1.9 GB against
# 2.3 GB free on the board, and a disk-full failure mid-run is worse than a
# missing point.
#
# Everything else, the differential counter method, the sha256 matching, the
# alternating target order and the sampler discipline, is inherited from
# homo_perf.sh unchanged, so the new rows are directly comparable to the old.
set -u

BIN=~/llm/llama.cpp/build/bin
WORK=~/homo_perf_llama
OUT=$WORK/homo_perf.jsonl
ADD=$WORK/homo_sweep_add.jsonl
PIDFILE=$WORK/.sampler_pids
mkdir -p "$WORK"
: > "$PIDFILE"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/run.log"; }

log "=== homo_perf llama-3.2-1b start ==="

R15=https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main

# tag|params|url. Llama-3.2-1B is 1.24B parameters, a third family and a
# different architecture from the Qwen2 models. Order alternates small and
# large so file size stays decorrelated from elapsed time.
TARGETS="Q4_0|1.24|$R15/Llama-3.2-1B-Instruct-Q4_0.gguf
Q8_0|1.24|$R15/Llama-3.2-1B-Instruct-Q8_0.gguf
Q4_K_M|1.24|$R15/Llama-3.2-1B-Instruct-Q4_K_M.gguf
Q6_K|1.24|$R15/Llama-3.2-1B-Instruct-Q6_K.gguf
Q5_K_M|1.24|$R15/Llama-3.2-1B-Instruct-Q5_K_M.gguf"

guard(){
  local avail disk
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  disk=$(df -m "$WORK" | awk 'NR==2{print $4}')
  [ "$avail" -lt 700 ] && { log "ABORT: MemAvailable ${avail}MB"; return 1; }
  [ "$disk" -lt 1500 ] && { log "ABORT: disk ${disk}MB"; return 1; }
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
  rm -f "$stop"
}

cleanup(){
  local p
  if [ -f "$PIDFILE" ]; then
    while read -r p; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done < "$PIDFILE"
  fi
  pkill -9 -x vcgencmd 2>/dev/null
}
trap cleanup EXIT INT TERM

PERF_T=2
PERF_R=3
NLO=128
NHI=256

fetch(){ local u=$1 d=$2 k
  for k in 1 2 3 4 5; do
    curl -sSL --retry 3 --retry-delay 5 -o "$d" "$u" 2>>"$WORK/run.log" \
      && [ -s "$d" ] && return 0
    log "    fetch attempt $k failed"; sleep 10
  done; return 1; }

while IFS='|' read -r tag params url; do
  [ -z "$tag" ] && continue
  guard || continue
  f="$WORK/m.gguf"
  log "[$params B / $tag] download"
  fetch "$url" "$f" || { log "  DOWNLOAD FAILED"; rm -f "$f"; continue; }
  bytes=$(stat -c %s "$f")
  if [ "$bytes" -lt 100000000 ]; then
    log "  BAD DOWNLOAD ($bytes bytes)"; rm -f "$f"; continue
  fi
  sha=$(sha256sum "$f" | cut -d' ' -f1)
  t_pre=$(vcgencmd measure_temp | tr -dc '0-9.')
  bg=$(pgrep -c -x vcgencmd 2>/dev/null); bg=${bg:-0}
  log "[$params B / $tag] bytes=$bytes sha=${sha:0:12} temp_pre=${t_pre}C stray=$bg"

  # thread curve, unperturbed, r=10
  "$BIN/llama-bench" -m "$f" -p 128 -n 128 -t 1,2,3,4 -r 10 -o json \
      > "$WORK/benchA_${params}_${tag}.json" 2>>"$WORK/run.log"
  # power, one sampler per thread count
  for th in 1 2 3 4; do
    pf="$WORK/pwr_${params}_${tag}_t${th}.txt"
    stop="$WORK/.stop_${params}_${tag}_t${th}"
    power_start "$pf" "$stop"
    "$BIN/llama-bench" -m "$f" -p 128 -n 128 -t "$th" -r 3 -o json \
        > /dev/null 2>>"$WORK/run.log"
    power_stop "$stop"
  done
  # differential counters, prefill disabled so only load and decode are counted
  for n in "$NLO" "$NHI"; do
    log "[$params B / $tag] perf n=$n t=$PERF_T"
    perf stat -e instructions,cycles,cache-misses \
        -o "$WORK/perf_${params}_${tag}_n${n}.txt" -- \
        "$BIN/llama-bench" -m "$f" -p 0 -n "$n" -r "$PERF_R" -t "$PERF_T" -o json \
        > "$WORK/benchP_${params}_${tag}_n${n}.json" 2>>"$WORK/run.log"
  done

  t_post=$(vcgencmd measure_temp | tr -dc '0-9.')
  log "[$params B / $tag] temp_post=${t_post}C leaked=$(pgrep -c -x vcgencmd 2>/dev/null)"

  python3 - "$WORK" "$tag" "$params" "full" "$bytes" "$sha" \
           "$t_pre" "$t_post" "$bg" "$PERF_T" "$PERF_R" "$NLO" "$NHI" \
           >> "$OUT" <<'PY'
import json, os, re, sys
work, tag, params, mode, byts, sha, t_pre, t_post, bg, pt, pr, nlo, nhi = sys.argv[1:14]
byts, pt, pr, nlo, nhi = int(byts), int(pt), int(pr), int(nlo), int(nhi)
def counters(n):
    p = "%s/perf_%s_%s_n%d.txt" % (work, params, tag, n)
    if not os.path.exists(p): return None
    txt, out = open(p).read(), {}
    for ev in ("instructions", "cycles", "cache-misses"):
        m = re.search(r"([\d,]+)\s+" + ev, txt)
        if m: out[ev] = int(m.group(1).replace(",", ""))
    return out or None
def tok_s(n):
    p = "%s/benchP_%s_%s_n%d.json" % (work, params, tag, n)
    try:
        for r in json.load(open(p)):
            if r.get("n_gen", 0) > 0: return r["avg_ts"]
    except Exception: pass
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

  python3 - "$WORK/benchA_${params}_${tag}.json" "$tag" "$params" "$bytes" \
           "$sha" "$WORK/pwr_${params}_${tag}" "$t_pre" "$t_post" "$bg" \
           >> "$ADD" <<'PY'
import json, sys, os
f, tag, params, byts, sha, pwrbase, t_pre, t_post, bg = sys.argv[1:10]
byts = int(byts)
def meanw(th):
    p = "%s_t%d.txt" % (pwrbase, th)
    if not os.path.exists(p): return None
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

  rm -f "$f"
  log "[$params B / $tag] done"
done <<< "$TARGETS"

log "=== DONE: $(wc -l < "$OUT") counter rows, $(wc -l < "$ADD") curve rows ==="
