#!/bin/bash
set -u
BIN=/c/llmpc/bin; MOD=/c/llmpc/models; WORK=/c/llmpc/sweep
OUT=$WORK/laptop_coreid_clean.jsonl
M=$MOD/ci.gguf
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/coreid.log"; }
URL=https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
fetch(){ local k; for k in $(seq 1 15); do
  curl -sSL --retry 3 --retry-delay 10 --connect-timeout 30 -o "$M" "$URL" 2>/dev/null \
    && [ -s "$M" ] && return 0; log "  fetch $k failed"; sleep 30; done; return 1; }
[ -s "$M" ] || fetch || { log "DOWNLOAD FAILED"; exit 1; }
b=$(stat -c %s "$M"); s=$(sha256sum "$M"|cut -d' ' -f1)
: > "$OUT"
log "=== clean per-core scan start (bytes=$b) ==="
for pass in up down; do
  if [ "$pass" = up ]; then order=$(seq 0 19); else order=$(seq 19 -1 0); fi
  log "pass=$pass"
  for i in $order; do
    m=$(python -c "print(hex(1 << $i))")
    "$BIN/llama-bench.exe" -m "$M" -p 128 -n 128 -t 1 -r 5 -C "$m" --cpu-strict 1 \
        -o json > "$WORK/ci_${pass}_$i.json" 2>>"$WORK/coreid.log"
    python - "$WORK/ci_${pass}_$i.json" "$pass" "$m" "$b" "$s" >> "$OUT" <<'PY'
import json, sys, statistics as st
f, arm, mask, byts, sha = sys.argv[1:6]
try: rows = json.load(open(f))
except Exception as e:
    print(json.dumps({"arm": "coreid_"+arm, "cpu_mask": mask, "error": str(e)})); raise SystemExit
for r in rows:
    s = r.get("samples_ts") or []
    print(json.dumps({
        "tag": "qwen0.5b", "params_B": 0.494, "arm": "coreid_"+arm,
        "cpu_mask": mask, "threads": r["n_threads"],
        "phase": "decode" if r.get("n_gen",0) > 0 else "prefill",
        "bytes": int(byts), "sha256": sha, "reps": len(s),
        "tok_s_mean": r["avg_ts"], "tok_s_stddev": r.get("stddev_ts"),
        "tok_s_median": st.median(s) if s else None, "samples_ts": s}))
PY
  done
done
rm -f "$M"
log "=== CLEAN CORE SCAN DONE ==="
