#!/bin/bash
# homo_sweep.sh - strengthen Paper 17's homogeneous-core control.
#
# WHY. CAL rejected Paper 17 3-0. Reviewer 3: "Just executing LLMs by varying
# configurations and reporting the final performance numbers is not at all
# interesting... The author needs to answer 'why' behind each and every
# observation." Reviewer 2: three repetitions is too few.
#
# The existing control (data/pi_decode_control.jsonl) shows the decline from
# peak to 4 threads GROWING with model size on homogeneous cores: 6.9% at 0.5B,
# 20.4% at 1.2B, 31.7% at 1.5B. That direction is the paper's argument, and it
# holds. But it rests on three models at r=3, and "model size" confounds two
# variables: parameter count and bytes streamed per token.
#
# THIS SEPARATES THEM. Fixed architecture (Qwen2.5-0.5B), fixed parameter count,
# sweeping only the bytes by walking the quantization ladder from Q2_K (415 MB)
# to Q8_0 (676 MB), then one 1.5B point to extend the byte range past 1 GB.
# If the 4-thread decline tracks bytes and not parameters, the saturation
# mechanism is isolated rather than asserted, which is the "why" the review
# asked for.
#
# Artifacts are the OFFICIAL published GGUFs, downloaded one at a time and
# deleted after benching so the 2 GB board's 4 GB of free disk is never at risk.
# Using the published files rather than locally requantized ones matters here:
# the files already on this board from July tie the embedding to the output head
# and therefore stream different bytes (see paper16 FINDING1_RESOLVED.md). Mixing
# them in would reintroduce exactly that confound.
#
# r=10 with standard deviations, threads 1..4, power sampled per thread count.
set -u

BIN=~/llm/llama.cpp/build/bin
WORK=~/homo_sweep
OUT=$WORK/homo_sweep.jsonl
mkdir -p "$WORK"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/run.log"; }

BUILD=$("$BIN/llama-bench" --help >/dev/null 2>&1; echo "d73c1d6-binary")
log "=== homo_sweep start (binary in $BIN) ==="

R0=https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main
R15=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main

# tag|params_B|url
#
# ORDER MATTERS AND THIS ORDER IS DELIBERATE. The quantity under test is how the
# 4-thread decline varies with bytes streamed. Walking the ladder monotonically
# from 415 MB to 1117 MB would make byte size perfectly collinear with elapsed
# time, so any slow drift over the ~100 minute campaign (board warming, DRAM
# refresh, page-cache state after a 1.1 GB download) would be indistinguishable
# from the effect we are trying to measure, and would inflate it in exactly the
# direction that flatters the hypothesis. Alternating small and large targets
# decorrelates size from time. bitfloor7 got the same protection by interleaving
# its two arms; this is the single-arm equivalent.
TARGETS="
Q2_K|0.5|$R0/qwen2.5-0.5b-instruct-q2_k.gguf
Q4_K_M|1.5|$R15/qwen2.5-1.5b-instruct-q4_k_m.gguf
Q3_K_M|0.5|$R0/qwen2.5-0.5b-instruct-q3_k_m.gguf
Q3_K_M|1.5|$R15/qwen2.5-1.5b-instruct-q3_k_m.gguf
Q4_0|0.5|$R0/qwen2.5-0.5b-instruct-q4_0.gguf
Q8_0|0.5|$R0/qwen2.5-0.5b-instruct-q8_0.gguf
Q4_K_M|0.5|$R0/qwen2.5-0.5b-instruct-q4_k_m.gguf
Q6_K|0.5|$R0/qwen2.5-0.5b-instruct-q6_k.gguf
Q5_K_M|0.5|$R0/qwen2.5-0.5b-instruct-q5_k_m.gguf
"

guard(){
  local avail disk
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  disk=$(df -m "$WORK" | awk 'NR==2{print $4}')
  [ "$avail" -lt 700 ] && { log "ABORT: MemAvailable ${avail}MB"; return 1; }
  [ "$disk" -lt 1600 ] && { log "ABORT: disk ${disk}MB"; return 1; }
  return 0
}

# 10 Hz PMIC rail sum.
#
# The sampler stops on a STOP FILE, not on a timeout and not on kill. The
# previous version backgrounded a subshell and killed its PID; that does not
# reach the vcgencmd or sleep the subshell is blocked in, and one such sampler
# survived six hours past its parent and perturbed a whole campaign. A loop that
# checks for its own stop file terminates itself even if the kill misses, and
# needs no guess at how long the run will take.
power_watch(){
  local out=$1 stop=$2
  rm -f "$stop"
  ( while [ ! -f "$stop" ]; do
      vcgencmd pmic_read_adc 2>/dev/null | awk '
        /current\(/{split($0,a,"="); c[$1]=a[2]+0}
        /volt\(/{split($0,a,"="); v[$1]=a[2]+0}
        END{p=0; for(k in c){ n=k; sub(/_A$/,"",n); vk=n"_V"; if(vk in v) p+=c[k]*v[vk]} print p}'
      sleep 0.1
    done ) > "$out" 2>/dev/null &
  echo $!
}

# Whatever happens, leave nothing sampling behind.
cleanup(){ touch "$WORK"/.stop_* 2>/dev/null; pkill -9 -f "pmic_read_adc" 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "$TARGETS" | while IFS='|' read -r tag params url; do
  [ -z "$tag" ] && continue
  guard || continue
  f="$WORK/m.gguf"
  log "[$params B / $tag] download"
  curl -sL -o "$f" "$url" || { log "  download FAILED"; rm -f "$f"; continue; }
  bytes=$(stat -c %s "$f")
  # a truncated or error-page download is small; refuse to bench it
  if [ "$bytes" -lt 100000000 ]; then
    log "  BAD DOWNLOAD ($bytes bytes), skipping"; rm -f "$f"; continue
  fi
  sha=$(sha256sum "$f" | cut -d' ' -f1)
  t_pre=$(vcgencmd measure_temp | tr -dc '0-9.')
  log "[$params B / $tag] bytes=$bytes temp_pre=${t_pre}C"

  # pass A: throughput, unperturbed, r=10 for tight error bars
  "$BIN/llama-bench" -m "$f" -p 128 -n 128 -t 1,2,3,4 -r 10 -o json \
      > "$WORK/benchA_${params}_${tag}.json" 2>>"$WORK/run.log"

  # pass B: power per thread count
  for th in 1 2 3 4; do
    pf="$WORK/pwr_${params}_${tag}_t${th}.txt"
    stop="$WORK/.stop_${params}_${tag}_t${th}"
    pid=$(power_watch "$pf" "$stop")
    "$BIN/llama-bench" -m "$f" -p 128 -n 128 -t "$th" -r 3 -o json \
        > /dev/null 2>>"$WORK/run.log"
    touch "$stop"; wait "$pid" 2>/dev/null; rm -f "$stop"
  done

  t_post=$(vcgencmd measure_temp | tr -dc "0-9.")
  log "[$params B / $tag] temp_post=${t_post}C"
  python3 - "$WORK/benchA_${params}_${tag}.json" "$tag" "$params" "$bytes" \
           "$sha" "$WORK/pwr_${params}_${tag}" "$t_pre" "$t_post" >> "$OUT" <<'PY'
import json, sys, os
f, tag, params, byts, sha, pwrbase, t_pre, t_post = sys.argv[1:9]
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
        "tok_s": r["avg_ts"], "stddev": r.get("stddev_ts"),
        "reps": len(r.get("samples_ts", [])),
        "mean_W": w,
        "mJ_per_tok": (w/r["avg_ts"]*1000) if (w and r["avg_ts"]) else None}))
PY
  rm -f "$f"
  log "[$params B / $tag] done"
done

log "=== DONE ==="
