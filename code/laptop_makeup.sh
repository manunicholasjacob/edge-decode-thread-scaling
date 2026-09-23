#!/bin/bash
# laptop_makeup.sh - recover the models a network outage cost, and re-run the
# headline model on a quiet machine.
#
# WHAT HAPPENED. During the main sweep, DNS stopped resolving huggingface.co for
# several minutes. The fetch helper retries five times, exhausted them on
# qwen1.5b and qwen3b, logged the failure and moved on, which is the behaviour
# it should have: a silent skip that leaves a log line beats a half-downloaded
# model benchmarked as if it were whole. But it leaves two of the five models
# missing, and the size-scaling law is the paper's actual claim, so they have to
# be made up.
#
# It also re-runs qwen0.5b. That model's sweep overlapped about six minutes of
# my own activity on this laptop (a git commit, process listings, an ssh
# session), and it is the model the headline number comes from. Two independent
# runs of the same configuration, one of them on a verified-quiet machine, is
# the only way to tell a real 35% cliff from a sampling artifact. They are
# written to separate files and compared rather than pooled.
#
# Retries are more patient here than in the main sweep: fifteen attempts with a
# longer wait, because a five-minute outage should not cost another model.
set -u

BIN=/c/llmpc/bin
MOD=/c/llmpc/models
WORK=/c/llmpc/sweep
mkdir -p "$MOD" "$WORK"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/makeup.log"; }

SWEEP_T=1,2,3,4,5,6,7,8,9,10,12,14,16,18,20
REPS=10
HF=https://huggingface.co

# tag|params|outfile|url
TARGETS="qwen1.5b|1.544|laptop_sweep.jsonl|$HF/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
qwen3b|3.086|laptop_sweep.jsonl|$HF/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
qwen0.5b|0.494|laptop_sweep_rerun.jsonl|$HF/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

fetch(){                      # url dest
  local u=$1 d=$2 k
  for k in $(seq 1 15); do
    if curl -sSL --retry 3 --retry-delay 10 --connect-timeout 30 -o "$d" "$u" \
         2>>"$WORK/makeup.log" && [ -s "$d" ]; then
      return 0
    fi
    log "    fetch attempt $k failed, waiting"
    sleep 30
  done
  return 1
}

log "=== laptop_makeup start ==="

while IFS='|' read -r tag params outfile url; do
  [ -z "$tag" ] && continue
  f="$MOD/mk_$tag.gguf"
  log "[$tag] download -> $outfile"
  fetch "$url" "$f" || { log "  DOWNLOAD FAILED after 15 attempts"; rm -f "$f"; continue; }
  bytes=$(stat -c %s "$f")
  if [ "$bytes" -lt 100000000 ]; then
    log "  BAD DOWNLOAD ($bytes bytes)"; rm -f "$f"; continue
  fi
  sha=$(sha256sum "$f" | cut -d' ' -f1)
  log "[$tag] bytes=$bytes sha=${sha:0:12} sweeping"

  "$BIN/llama-bench.exe" -m "$f" -p 128 -n 128 -t "$SWEEP_T" -r "$REPS" -o json \
      > "$WORK/bench_mk_$tag.json" 2>>"$WORK/makeup.log"

  python - "$WORK/bench_mk_$tag.json" "$tag" "$params" "$bytes" "$sha" "$url" \
      >> "$WORK/$outfile" <<'PY'
import json, sys, statistics as st
f, tag, params, byts, sha, url = sys.argv[1:7]
try:
    rows = json.load(open(f))
except Exception as e:
    print(json.dumps({"tag": tag, "error": str(e)})); raise SystemExit
for r in rows:
    s = r.get("samples_ts") or []
    print(json.dumps({
        "tag": tag, "params_B": float(params), "arm": "default",
        "cpu_mask": "none", "threads": r["n_threads"],
        "phase": "decode" if r.get("n_gen", 0) > 0 else "prefill",
        "bytes": int(byts), "sha256": sha, "source_url": url,
        "build": r.get("build_commit"), "reps": len(s),
        "tok_s_mean": r["avg_ts"], "tok_s_stddev": r.get("stddev_ts"),
        "tok_s_median": st.median(s) if s else None,
        "tok_s_min": min(s) if s else None,
        "tok_s_max": max(s) if s else None,
        "samples_ts": s}))
PY

  rm -f "$f"
  log "[$tag] done"
done <<< "$TARGETS"

log "=== MAKEUP DONE ==="
