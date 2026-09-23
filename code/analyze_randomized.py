#!/usr/bin/env python3
"""What the randomised sweep actually says, past the medians.

The medians alone read like a gentle curve: throughput rises to a peak at
sixteen threads and falls 20.8% by twenty. But the per-cell ranges do not look
like that at all. At twenty threads the samples span 30.4 to 112.2 tok/s, and at
eighteen they span 44.4 to 122.2, while at nine threads they span 114.6 to 117.0.
A median summarising the first of those is describing a mixture, not a value.

So this reports the shape of each cell's distribution rather than one number per
cell, and asks a different question: does oversubscription make decode SLOWER,
or does it make decode UNRELIABLE? Those imply different deployment advice, and
only the second is consistent with samples that are mostly fine and occasionally
catastrophic.

It also does the drift check properly. The position medians printed by the sweep
script mix thread counts, because position 5 holds a different thread count in
each pass, so with five passes each position median is a blend of five unrelated
cells and tells you very little. The right test normalises each sample by its own
thread count's median first, then asks whether position predicts the residual.
"""
import collections
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "data", "randomized_sweep.jsonl")
PCORES = 6


def pct(vals, p):
    s = sorted(vals)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


rows = [json.loads(l) for l in open(SRC) if l.strip()]
dec = [r for r in rows if r.get("phase") == "decode" and "error" not in r]
if not dec:
    raise SystemExit("no decode rows in %s" % SRC)

by = collections.defaultdict(list)
samples = []                                   # (threads, position, value)
for r in dec:
    by[r["threads"]].extend(r["samples_ts"])
    for v in r["samples_ts"]:
        samples.append((r["threads"], r["position_in_pass"], v))

print("randomised sweep: %d passes, %d cells, %d samples"
      % (len(set(r["pass"] for r in dec)), len(dec), len(samples)))
print()
print("%5s %8s %8s %8s %8s %8s %7s %7s"
      % ("t", "median", "p25", "p75", "min", "max", "CV%", "min/med"))
for t in sorted(by):
    v = by[t]
    m = st.median(v)
    cv = 100 * st.pstdev(v) / st.mean(v) if len(v) > 1 else 0
    print("%5d %8.1f %8.1f %8.1f %8.1f %8.1f %7.1f %7.2f"
          % (t, m, pct(v, 25), pct(v, 75), min(v), max(v), cv, min(v) / m))

peak_t = max(by, key=lambda t: st.median(by[t]))
peak = st.median(by[peak_t])
top = max(by)
print()
print("peak            %.1f tok/s at t=%d" % (peak, peak_t))
print("at t=%-2d         %.1f  (median fall %.1f%%)"
      % (top, st.median(by[top]), 100 * (1 - st.median(by[top]) / peak)))
print("at P-core count %.1f at t=%d  (cost of the paper's rule: %.1f%%)"
      % (st.median(by[PCORES]), PCORES,
         100 * (1 - st.median(by[PCORES]) / peak)))

# Slower, or just unreliable? Compare the typical run against the bad tail.
print()
print("is oversubscription slower, or unreliable?")
for t in sorted(by):
    if t < 12:
        continue
    v = by[t]
    print("   t=%-2d  p75 %6.1f (%.0f%% of peak p75)   p25 %6.1f   worst %6.1f (%.0f%% of peak)"
          % (t, pct(v, 75), 100 * pct(v, 75) / pct(by[peak_t], 75),
             pct(v, 25), min(v), 100 * min(v) / peak))

# Drift, done properly: normalise each sample by its thread count's median,
# then see whether position predicts the residual.
med = {t: st.median(by[t]) for t in by}
res = collections.defaultdict(list)
for t, p, v in samples:
    res[p].append(v / med[t])
print()
print("drift check (each sample normalised by its own thread count's median):")
xs, ys = [], []
for p in sorted(res):
    r = st.median(res[p])
    xs.append(float(p))
    ys.append(r)
    print("   position %2d  relative %.3f  n=%d" % (p, r, len(res[p])))
mx, my = st.mean(xs), st.mean(ys)
den = sum((x - mx) ** 2 for x in xs)
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0
print("   slope %+.4f per position; across %d positions that is %+.1f%%"
      % (slope, len(xs), 100 * slope * (max(xs) - min(xs))))
print("   (a clean randomisation leaves this near zero; the sequential sweep's")
print("    equivalent was a 14.3% session drift on identical work)")
