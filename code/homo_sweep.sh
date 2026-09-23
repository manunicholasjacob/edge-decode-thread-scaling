#!/bin/bash
# homo_sweep.sh (v2) - strengthen Paper 17's homogeneous-core control.
#
# WHY. CAL rejected Paper 17 3-0. Reviewer 3: "Just executing LLMs by varying
# configurations and reporting the final performance numbers is not at all
# interesting... The author needs to answer 'why' behind each and every
# observation." Reviewer 2: three repetitions is too few.
#
# The existing control shows the peak-to-4-thread decline GROWING with model
# size on homogeneous cores: 6.9% at 0.5B, 20.4% at 1.2B, 31.7% at 1.5B. That
# direction is the paper's argument and it holds. But it rests on three models
# at r=3, and "model size" confounds parameter count with bytes streamed.
#
# THIS SEPARATES THEM by holding the architecture fixed and walking the
# quantization ladder, plus two 1.5B points to extend the byte range past 1 GB.
#
# ORDER IS DELIBERATE. Walking the ladder monotonically would make byte size
# collinear with elapsed time, so board warming would be indistinguishable from
# the effect under test and would inflate it in the direction that flatters the
# hypothesis. Targets alternate small and large.
#
# SAMPLER DISCIPLINE (the v1 defect this version exists to fix). v1 leaked its
# power samplers: 36 were still spinning hours later and the board's load
# average reached 32. Two bugs compounded.
#   1. `pid=$(power_watch ...)` ran the function inside a COMMAND SUBSTITUTION
#      subshell, so the backgrounded loop was a grandchild of this shell.
#      `wait "$pid"` therefore returned immediately instead of blocking, and the
#      `rm -f "$stop"` that followed deleted the stop file before the loop's next
#      0.1 s poll could see it. The loop then never terminated.
#   2. `pkill -f pmic_read_adc` could not clean up after it, because a forked
#      shell subshell inherits the PARENT's argv: those loops appear in ps as
#      "bash homo_sweep.sh", which does not match that pattern.
# Fixed three independent ways: the sampler is started as a direct child so its
# PID is real and killable, it carries a hard iteration cap as a backstop, and
# every PID is journalled to a file that cleanup kills explicitly. Each target
# also records how many stray samplers were alive when it began, so a future
# leak shows up in the data instead of having to be inferred.
set -u

BIN=~/llm/llama.cpp/build/bin
WORK=~/homo_sweep
OUT=$WORK/homo_sweep.jsonl
PIDFILE=$WORK/.sampler_pids
mkdir -p "$WORK"
: > "$PIDFILE"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/run.log"; }

log "=== homo_sweep v2 start (binary in $BIN) ==="

R0=https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main
R15=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main

TARGETS="Q2_K|0.5|$R0/qwen2.5-0.5b-instruct-q2_k.gguf
Q4_K_M|1.5|$R15/qwen2.5-1.5b-instruct-q4_k_m.gguf
Q3_K_M|0.5|$R0/qwen2.5-0.5b-instruct-q3_k_m.gguf
Q3_K_M|1.5|$R15/qwen2.5-1.5b-instruct-q3_k_m.gguf
Q4_0|0.5|$R0/qwen2.5-0.5b-instruct-q4_0.gguf
Q8_0|0.5|$R0/qwen2.5-0.5b-instruct-q8_0.gguf
Q4_K_M|0.5|$R0/qwen2.5-0.5b-instruct-q4_k_m.gguf
Q6_K|0.5|$R0/qwen2.5-0.5b-instruct-q6_k.gguf
Q5_K_M|0.5|$R0/qwen2.5-0.5b-instruct-q5_k_m.gguf"

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

# `while read` over a here-string, NOT a pipe: a pipe runs the loop body in a
# subshell, where PWPID assignments would not survive back to this shell and the
# v1 bug would reappear in a new form.
while IFS='|' read -r tag params url; do
  [ -z "$tag" ] && continue
  guard || continue
  f="$WORK/m.gguf"
  log "[$params B / $tag] download"
  curl -sL -o "$f" "$url" || { log "  download FAILED"; rm -f "$f"; continue; }
  bytes=$(stat -c %s "$f")
  if [ "$bytes" -lt 100000000 ]; then
    log "  BAD DOWNLOAD ($bytes bytes), skipping"; rm -f "$f"; continue
  fi
  sha=$(sha256sum "$f" | cut -d' ' -f1)
  t_pre=$(vcgencmd measure_temp | tr -dc '0-9.')
  bg=$(pgrep -c -x vcgencmd 2>/dev/null); bg=${bg:-0}
  log "[$params B / $tag] bytes=$bytes temp_pre=${t_pre}C stray_samplers=$bg"

  # pass A: throughput, UNPERTURBED, r=10 for tight error bars
  "$BIN/llama-bench" -m "$f" -p 128 -n 128 -t 1,2,3,4 -r 10 -o json \
      > "$WORK/benchA_${params}_${tag}.json" 2>>"$WORK/run.log"

  # pass B: power, one sampler per thread count, each stopped before the next
  for th in 1 2 3 4; do
    pf="$WORK/pwr_${params}_${tag}_t${th}.txt"
    stop="$WORK/.stop_${params}_${tag}_t${th}"
    power_start "$pf" "$stop"
    "$BIN/llama-bench" -m "$f" -p 128 -n 128 -t "$th" -r 3 -o json \
        > /dev/null 2>>"$WORK/run.log"
    power_stop "$stop"
  done

  t_post=$(vcgencmd measure_temp | tr -dc '0-9.')
  log "[$params B / $tag] temp_post=${t_post}C leaked_after=$(pgrep -c -x vcgencmd 2>/dev/null)"

  python3 - "$WORK/benchA_${params}_${tag}.json" "$tag" "$params" "$bytes" \
           "$sha" "$WORK/pwr_${params}_${tag}" "$t_pre" "$t_post" "$bg" \
           >> "$OUT" <<'PY'
import json, sys, os
f, tag, params, byts, sha, pwrbase, t_pre, t_post, bg = sys.argv[1:10]
byts = int(byts)
def meanw(th):
    p = "%s_t%d.txt" % (pwrbase, th)
    if not os.path.exists(p): return None
    v = [float(x) for x in open(p) if x.strip()]
    return sum(v)/len(v) if v else None
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
        "mJ_per_tok": (w/r["avg_ts"]*1000) if (w and r["avg_ts"]) else None}))
PY
  rm -f "$f"
  log "[$params B / $tag] done"
done <<< "$TARGETS"

log "=== DONE ==="
