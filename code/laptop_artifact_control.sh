#!/bin/bash
# laptop_artifact_control.sh - separate "the cliff was a sampling artifact" from
# "the cliff was measured on a different file".
#
# WHAT THE COMPARISON TURNED UP. The re-measurement gives a 15.2% fall on the
# 0.5B model where the paper claims 35.1%. The obvious suspects were the binary
# and the sampling. The binary is not it: both campaigns record llama.cpp build
# 0e4a03622, because the archived Windows build on this machine IS the one the
# August campaign used. What differs is the FILE.
#
#   August campaign : 391,859,712 bytes (373.7 MiB), local C:/llmpc/models/
#   re-measurement  : 491,400,032 bytes (468.6 MiB), Qwen's published repo
#
# Both are labelled Q4_K_M, for the same model, and they differ by 99.5 MB.
# That is 25% more bytes streamed per token in the file I measured, under a
# label that promises they are the same thing. Paper 16 is a whole paper about
# exactly this, and the gap is about the size of an untied output head: Qwen2.5
# 0.5B has a 151936 x 896 embedding, which as a K-quantized output tensor is
# roughly 100 MB, and Paper 16 established that locally requantized files tie
# the embedding to the output head while published ones do not (290 tensors
# against 291).
#
# THIS MATTERS BECAUSE THE PAPER'S OWN MECHANISM PREDICTS THE DIRECTION. The
# cliff depth is supposed to be gated by how much bandwidth headroom a model
# leaves: a model that streams more bytes sits closer to saturation and should
# cliff LESS. The file I measured streams 25% more and cliffs less. So the
# discrepancy is not obviously an error in either campaign; it may be the
# mechanism working.
#
# THE EXPERIMENT. Build the tied-embedding artifact locally from the official
# FP16, exactly as llama-quantize would have in August, and sweep it beside the
# published one on the same machine, the same binary, the same day. Then:
#
#   * if the local artifact reproduces ~35% and the published one gives ~15%,
#     the paper's number is right for the file it measured and the paper has an
#     artifact-provenance problem, not a sampling problem;
#   * if both give ~15%, the 35% was a sampling artifact of single draws;
#   * if the local artifact's byte count comes out at 391,859,712, that also
#     confirms what the August file was, which nothing else can now recover
#     because the file itself is gone from this machine.
#
# The byte count is the cheapest and sharpest part of this: it is a single
# number that either matches the August record exactly or does not.
set -u

BIN=/c/llmpc/bin
MOD=/c/llmpc/models
WORK=/c/llmpc/sweep
OUT=$WORK/laptop_artifact_control.jsonl
mkdir -p "$MOD" "$WORK"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/artifact.log"; }

SWEEP_T=1,2,3,4,5,6,7,8,9,10,12,14,16,18,20
REPS=10
HF=https://huggingface.co
FP16=$MOD/ac_fp16.gguf
AUGUST_BYTES=391859712

fetch(){ local u=$1 d=$2 k
  for k in $(seq 1 15); do
    curl -sSL --retry 3 --retry-delay 10 --connect-timeout 30 -o "$d" "$u" \
      2>>"$WORK/artifact.log" && [ -s "$d" ] && return 0
    log "    fetch attempt $k failed"; sleep 30
  done; return 1; }

sweep(){                     # tag file bytes sha note
  local tag=$1 f=$2 bytes=$3 sha=$4 note=$5
  log "[$tag] sweeping ($bytes bytes)"
  "$BIN/llama-bench.exe" -m "$f" -p 128 -n 128 -t "$SWEEP_T" -r "$REPS" -o json \
      > "$WORK/bench_ac_$tag.json" 2>>"$WORK/artifact.log"
  python - "$WORK/bench_ac_$tag.json" "$tag" "$bytes" "$sha" "$note" >> "$OUT" <<'PY'
import json, sys, statistics as st
f, tag, byts, sha, note = sys.argv[1:6]
try:
    rows = json.load(open(f))
except Exception as e:
    print(json.dumps({"tag": tag, "error": str(e)})); raise SystemExit
for r in rows:
    s = r.get("samples_ts") or []
    print(json.dumps({
        "tag": tag, "provenance": note, "params_B": 0.494,
        "threads": r["n_threads"],
        "phase": "decode" if r.get("n_gen", 0) > 0 else "prefill",
        "bytes": int(byts), "sha256": sha, "build": r.get("build_commit"),
        "reps": len(s), "tok_s_mean": r["avg_ts"],
        "tok_s_stddev": r.get("stddev_ts"),
        "tok_s_median": st.median(s) if s else None,
        "tok_s_min": min(s) if s else None, "tok_s_max": max(s) if s else None,
        "samples_ts": s}))
PY
  log "[$tag] done"
}

log "=== artifact control start ==="

log "fetching official FP16 source"
fetch "$HF/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf" "$FP16" \
  || { log "FP16 DOWNLOAD FAILED"; exit 1; }

LOCAL=$MOD/ac_local_q4km.gguf
log "quantizing FP16 -> Q4_K_M locally (this is what August most likely had)"
"$BIN/llama-quantize.exe" "$FP16" "$LOCAL" Q4_K_M >> "$WORK/quantize.log" 2>&1
if [ ! -s "$LOCAL" ]; then
  log "QUANTIZE FAILED, see quantize.log"; rm -f "$FP16"; exit 1
fi
lb=$(stat -c %s "$LOCAL"); lsha=$(sha256sum "$LOCAL" | cut -d' ' -f1)
log "local artifact: $lb bytes (August record: $AUGUST_BYTES)"
if [ "$lb" -eq "$AUGUST_BYTES" ]; then
  log "  *** EXACT MATCH with the August file size ***"
else
  log "  differs from the August file by $((lb - AUGUST_BYTES)) bytes"
fi
rm -f "$FP16"

sweep "local_q4km" "$LOCAL" "$lb" "$lsha" "locally requantized from official FP16"
rm -f "$LOCAL"

PUB=$MOD/ac_pub_q4km.gguf
log "fetching the published Q4_K_M for the paired arm"
fetch "$HF/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" "$PUB" \
  || { log "PUBLISHED DOWNLOAD FAILED"; exit 1; }
pb=$(stat -c %s "$PUB"); psha=$(sha256sum "$PUB" | cut -d' ' -f1)
sweep "published_q4km" "$PUB" "$pb" "$psha" "official published GGUF"
rm -f "$PUB"

log "=== ARTIFACT CONTROL DONE ==="
python - "$OUT" <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = collections.defaultdict(dict)
for r in rows:
    if r.get("phase") == "decode" and "error" not in r:
        by[r["tag"]][r["threads"]] = r["tok_s_median"]
for tag, d in by.items():
    if not d:
        continue
    pk = max(d, key=d.get)
    hi = max(d)
    print("%-16s peak %.1f tok/s at t=%d ; t=%d %.1f ; fall %.1f%%"
          % (tag, d[pk], pk, hi, d[hi], 100 * (1 - d[hi] / d[pk])))
print("paper claims: peak 86.4 at t=8, 56.4 at t=20, fall 35.1%")
PY
