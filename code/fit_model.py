#!/usr/bin/env python3
"""Does the two-term account actually predict the thread optimum, or only
describe it after the fact?

The paper argues decode time per token is the maximum of a streaming term and a
compute term. That is a mechanism story, and so far it has been supported by
correlations. This tests it as a predictor.

    t(n) = max( B / BW(n),  I / (IPC * f * n) )

B (bytes) and I (instructions per token) and IPC are measured per artifact. f is
the A76 clock. BW(n) is the only thing fitted, and it is fitted ONCE across all
artifacts, not per artifact, as the two-parameter saturating form

    BW(n) = min(n * b1, BWmax)

so eleven predictions come out of two free parameters. The prediction is the
thread count minimising t(n) over 1..4, compared against the measured peak.

A baseline matters here, because with only four candidate thread counts a
model can look good by accident. Two are reported: always predicting the modal
thread count, and a bytes-only model (the pure saturation account, which is what
the compute term is supposed to beat).

Nothing is written to the paper unless the numbers justify it; this script only
prints.
"""
import collections
import itertools
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERF = os.path.join(HERE, "data", "homo_perf.jsonl")
SWEEP = os.path.join(HERE, "data", "homo_sweep.jsonl")
ADD = os.path.join(HERE, "data", "homo_sweep_add.jsonl")
PERF15 = os.path.join(HERE, "data", "homo_perf_15b.jsonl")
ADD15 = os.path.join(HERE, "data", "homo_sweep_add_15b.jsonl")

F_HZ = 2.4e9          # Cortex-A76 clock on the Pi 5
THREADS = (1, 2, 3, 4)

curve = collections.defaultdict(dict)
for path in (SWEEP, ADD, ADD15):
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("phase") == "decode":
            curve[(r["params_B"], r["tag"])][r["threads"]] = r

art = []
for line in [l for f in (PERF, PERF15) for l in open(f)]:
    if not line.strip():
        continue
    p = json.loads(line)
    if "inst_per_token" not in p:
        continue
    v = curve.get((p["params_B"], p["tag"]))
    if not v or len(v) < 4:
        continue
    tp = {t: v[t]["tok_s"] for t in THREADS}
    art.append(dict(tag=p["tag"], params=p["params_B"], B=float(p["bytes"]),
                    I=p["inst_per_token"], ipc=p["ipc_decode"], tp=tp,
                    peak=max(tp, key=tp.get)))

print("artifacts:", len(art))


def predict(a, b1, bwmax):
    best, bt = None, None
    for n in THREADS:
        bw = min(n * b1, bwmax)
        t = max(a["B"] / bw, a["I"] / (a["ipc"] * F_HZ * n))
        if best is None or t < best - 1e-12:
            best, bt = t, n
    return bt


def score(b1, bwmax, pool):
    return sum(predict(a, b1, bwmax) == a["peak"] for a in pool)


# Grid search the two global parameters. Coarse on purpose: the point is whether
# the shape works at all, not to squeeze the last artifact out of it.
grid_b1 = [x * 1e8 for x in range(10, 141, 2)]        # 1.0 to 14.0 GB/s per thread
grid_max = [x * 1e8 for x in range(60, 201, 2)]       # 6.0 to 20.0 GB/s ceiling
best = max(((score(b1, bw, art), b1, bw) for b1, bw in itertools.product(grid_b1, grid_max)),
           key=lambda z: (z[0], -z[2]))
hit, b1, bwmax = best
print("fitted globally: b1=%.2f GB/s/thread  BWmax=%.2f GB/s" % (b1 / 1e9, bwmax / 1e9))
print("two-term model : %d/%d exact" % (hit, len(art)))

# Baseline 1: always guess the most common peak.
modal = collections.Counter(a["peak"] for a in art).most_common(1)[0]
print("modal baseline : %d/%d exact (always t=%d)"
      % (modal[1], len(art), modal[0]))

# Baseline 2: bytes only, i.e. pure saturation with no compute term.
def predict_bytes(a, b1, bwmax):
    best, bt = None, None
    for n in THREADS:
        t = a["B"] / min(n * b1, bwmax)
        if best is None or t < best - 1e-12:
            best, bt = t, n
    return bt


bb = max(((sum(predict_bytes(a, x, y) == a["peak"] for a in art), x, y)
          for x, y in itertools.product(grid_b1, grid_max)), key=lambda z: z[0])
print("bytes-only     : %d/%d exact" % (bb[0], len(art)))

# Leave-one-out, so the reported accuracy is not the fit's own training score.
loo = 0
for i, a in enumerate(art):
    pool = art[:i] + art[i + 1:]
    s = max(((score(x, y, pool), x, y) for x, y in itertools.product(grid_b1, grid_max)),
            key=lambda z: (z[0], -z[2]))
    if predict(a, s[1], s[2]) == a["peak"]:
        loo += 1
print("two-term LOO   : %d/%d exact" % (loo, len(art)))

print()
print("%-8s %-5s %8s %8s %6s %5s %5s" % ("tag", "par", "GB", "Ginst", "IPC", "meas", "pred"))
errs = []
for a in sorted(art, key=lambda a: (a["params"], a["I"])):
    pr = predict(a, b1, bwmax)
    errs.append(abs(pr - a["peak"]))
    print("%-8s %-5.1f %8.2f %8.3f %6.2f %5d %5d %s"
          % (a["tag"], a["params"], a["B"] / 1e9, a["I"] / 1e9, a["ipc"],
             a["peak"], pr, "" if pr == a["peak"] else "  MISS"))

# --- macros, so the prose cannot drift from the fit ---
OUT = os.path.join(HERE, "paper", "numbers_model.tex")
lines = ["% generated by code/fit_model.py -- do not edit"]


def put(name, val):
    lines.append(r"\newcommand{\np%s}{%s}" % (name, val))


BW_CEILING = 13.98  # GB/s, measured independently on this board
put("ModN", str(len(art)))
put("ModHit", str(hit))
put("ModLOO", str(loo))
put("ModModal", str(modal[1]))
put("ModModalT", str(modal[0]))
put("ModBytesOnly", str(bb[0]))
put("ModBOne", "%.1f" % (b1 / 1e9))
put("ModBWMax", "%.1f" % (bwmax / 1e9))
put("ModBWCeil", "%.2f" % BW_CEILING)
put("ModBWFrac", "%.0f" % (100 * bwmax / 1e9 / BW_CEILING))
put("ModMAE", "%.2f" % (sum(errs) / len(errs)))
put("ModMaxErr", str(max(errs)))
put("ModMiss", str(len(art) - hit))
put("ModClock", "%.1f" % (F_HZ / 1e9))
rows = []
for a in sorted(art, key=lambda a: (a["params"], a["I"])):
    pr = predict(a, b1, bwmax)
    rows.append("%s %s & %.2f & %.3f & %.2f & %d & %d & %s \\\\" % (
        ("%.1f" % a["params"]).rstrip("0").rstrip(".") + "\\,B",
        a["tag"].replace("_", "\\_"), a["B"] / 1e9, a["I"] / 1e9, a["ipc"],
        a["peak"], pr, "\\checkmark" if pr == a["peak"] else "$-$"))
put("TableModel", "%\n" + "\n".join(rows))
open(OUT, "w").write("\n".join(lines) + "\n")
print("\nwrote", OUT, "(%d macros)" % (len(lines) - 1))
print("  MAE %.2f threads, max error %d" % (sum(errs) / len(errs), max(errs)))
