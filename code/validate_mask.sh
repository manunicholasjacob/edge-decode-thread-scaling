#!/bin/bash
# validate_mask.sh - does llama.cpp's --cpu-mask actually bind threads on Windows?
#
# THIS MUST PASS BEFORE THE PLACEMENT EXPERIMENT MEANS ANYTHING. Phase B pinned
# one thread to each of the twenty logical processors in turn and got
# 35 to 41 tok/s on nineteen of them. On an i7-12700H, six Golden Cove cores
# beside eight Gracemont cores, single-thread compute-bound prefill should NOT
# be flat: an E-core should be visibly slower. Flat is what you see when the
# mask is parsed, recorded in the JSON, and then ignored.
#
# An earlier pilot did read 25.5 tok/s on bit 12 while bit 18 read 38.5, which
# looked like exactly the P/E split expected. The ten-repetition scan does not
# reproduce it: bit 12 now reads 40.6. So the pilot's low value was noise, and
# the flat scan is the result that stands.
#
# If the mask does not bind, a placement experiment built on it would compare
# arms that are physically identical, return a null, and the null would be an
# artifact of the instrument rather than a fact about the hardware. That is a
# worse outcome than having no placement experiment, so the instrument gets
# tested first and separately.
#
# THE TEST. Oversubscribe a single logical processor. Eight threads confined to
# one core must be far slower than eight threads on the whole machine, because
# they are time-slicing one core. If the two are close, the mask is not binding.
# This does not depend on knowing which core is which, and it does not depend on
# the P/E distinction at all, which is why it can settle the question that the
# per-core scan could not.
#
# Interpretation:
#   ratio (masked/unmasked) near 1/8      -> mask binds; placement study is valid
#   ratio near 1                          -> mask does NOT bind; use OS affinity
#   anything between                      -> partial binding; do not build on it
#
# Run only when nothing else is on the machine.
set -u

# Two finishers may be armed (an earlier chain and the later combined one).
# Only one may run a benchmark, or they contend and both results are worthless.
LOCK=/c/llmpc/.validate.lock
if ! ( set -o noclobber; echo "$$" > "$LOCK" ) 2>/dev/null; then
  echo "validate_mask: another instance holds $LOCK, skipping" >&2
  exit 0
fi
trap 'rm -f "$LOCK"' EXIT

BIN=/c/llmpc/bin
MOD=/c/llmpc/models
WORK=/c/llmpc/sweep
M="$MOD/validate.gguf"
OUT=$WORK/mask_validation.jsonl
mkdir -p "$MOD" "$WORK"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/validate.log"; }

URL=https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
if [ ! -s "$M" ]; then
  log "download"
  curl -sSL --retry 3 --retry-delay 5 -o "$M" "$URL" || { log "DOWNLOAD FAILED"; exit 1; }
fi

run(){                      # label threads maskargs...
  local label=$1 th=$2; shift 2
  "$BIN/llama-bench.exe" -m "$M" -p 128 -n 128 -t "$th" -r 5 "$@" -o json \
      > "$WORK/val_$label.json" 2>>"$WORK/validate.log"
  python - "$WORK/val_$label.json" "$label" "$th" "$*" >> "$OUT" <<'PY'
import json, sys, statistics as st
f, label, th, args = sys.argv[1:5]
try:
    rows = json.load(open(f))
except Exception as e:
    print(json.dumps({"label": label, "error": str(e)})); raise SystemExit
for r in rows:
    s = r.get("samples_ts") or []
    print(json.dumps({
        "label": label, "threads": int(th), "args": args,
        "phase": "decode" if r.get("n_gen", 0) > 0 else "prefill",
        "cpu_mask": r.get("cpu_mask"), "cpu_strict": r.get("cpu_strict"),
        "tok_s_mean": r["avg_ts"], "tok_s_median": st.median(s) if s else None,
        "samples_ts": s}))
PY
  log "  $label done"
}

log "=== mask validation start ==="
: > "$OUT"

# The decisive pair: same eight threads, one confined to a single logical
# processor, one free. Then the same at one thread as a floor reference.
run free8   8
run pin8    8 -C 0x1 --cpu-strict 1
run free1   1
run pin1    1 -C 0x1 --cpu-strict 1
# Two cores, as a middle point: should be about twice pin1 if binding works.
run pin8two 8 -C 0x3 --cpu-strict 1

log "=== DONE ==="
python - "$OUT" <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
v = collections.defaultdict(dict)
for r in rows:
    if "error" not in r:
        v[r["phase"]][r["label"]] = r["tok_s_median"] or r["tok_s_mean"]
for phase in ("prefill", "decode"):
    d = v.get(phase, {})
    if "free8" not in d or "pin8" not in d:
        continue
    ratio = d["pin8"] / d["free8"]
    print("%-8s free8=%7.2f  pin8(1 core)=%7.2f  ratio=%.3f  pin1=%7.2f  pin8two=%7.2f"
          % (phase, d["free8"], d["pin8"], ratio, d.get("pin1", float("nan")),
             d.get("pin8two", float("nan"))))
    if ratio < 0.4:
        print("         -> mask BINDS. Placement experiment is valid.")
    elif ratio > 0.8:
        print("         -> mask does NOT bind. Use OS process affinity instead.")
    else:
        print("         -> PARTIAL/UNCLEAR. Do not build a claim on this.")
PY
