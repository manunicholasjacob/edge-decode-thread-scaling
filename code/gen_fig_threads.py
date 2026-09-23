#!/usr/bin/env python3
"""Figure 1, driven entirely by the released JSONL.

REPLACES code/gen_fig_cliff.py, which hardcoded its laptop numbers in a dict at
the top of the file. Those numbers had drifted from the paper's own table: the
figure plotted 90.2 tok/s for the 0.5B model at eight threads where the table
said 86.4, and CAL's referees noticed the text and the figure disagreeing. A
figure that carries its own copy of the data will eventually disagree with the
table, so this one reads the same files the tables do and there is nowhere for a
second copy to live.

WHAT THE FIGURE NOW SHOWS, which is not what it used to show. The old panel
plotted one line per model, normalised to peak, to display a cliff. The finding
is no longer a cliff: under the randomised protocol throughput rises past the
P-core boundary to a peak well inside E-core territory, and what goes wrong
beyond it is that the LOWER TAIL collapses while the upper quartile stays near
peak. A line of medians hides exactly that, which is how the effect stayed
hidden. So the laptop panels draw the interquartile band and the minimum
alongside the median.

THE SIZE CONTRAST IS NOW PART OF THE FIGURE. Left, the 0.5B model, whose lower
tail falls to a quarter of peak. Centre, the 7.6B model measured under the same
protocol, whose band stays narrow because the machine is already at 97% of its
DRAM ceiling there. Right, the Pi, where the same protocol produces a curve
tight enough that the band is barely visible.

WHICH FILES IT READS, and why that had to be fixed. This script used to look
only for data/randomized_sweep_sizes.jsonl for the non-0.5B models. That file
does exist and holds the 1.2B, 1.5B and 3B sweeps, but the 7B shipped
separately as data/randomized_sweep_7b.jsonl, so the 7B was silently absent
from the figure. Both are read now.

AND IT APPLIES THE SAME STALL EXCLUSION AS THE ANALYSIS. The machine stalled
during the 7B run and seven cells are contaminated. The rule that removes them
lives in code/analyze_headroom_sizes.py and is imported here rather than
restated, because a figure filtering its data differently from the tables is
the same defect as a figure carrying its own copy of the data.

Type-42 fonts, because several target venues reject Type-3.
"""
import argparse
import collections
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "code"))
from analyze_headroom_sizes import stall_excluded          # noqa: E402

RAND = os.path.join(HERE, "data", "randomized_sweep.jsonl")
SIZES = os.path.join(HERE, "data", "randomized_sweep_sizes.jsonl")
SEVENB = os.path.join(HERE, "data", "randomized_sweep_7b.jsonl")
PI = os.path.join(HERE, "data", "homo_sweep.jsonl")
OUT = os.path.join(HERE, "paper", "fig_cliff.pdf")
PCORES = 6
BW_CEILING_GBS = 53.9


def pct(vals, p):
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load_laptop():
    """Per model, per thread count, every surviving decode sample.

    Each file is filtered on its own, because the exclusion threshold is a
    median duration within a run and does not transfer between runs.
    """
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    nbytes, dropped = {}, {}
    for path in (RAND, SIZES, SEVENB):
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        drop, _, _, _ = stall_excluded(rows)
        for r in rows:
            if "error" in r or r.get("phase") != "decode":
                continue
            if (r["pass"], r["position_in_pass"]) in drop:
                continue
            d[r["tag"]][r["threads"]].extend(r["samples_ts"])
            nbytes[r["tag"]] = r["bytes"]
        dropped[os.path.basename(path)] = len(drop)
    return d, nbytes, dropped


def load_pi():
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    if not os.path.exists(PI):
        return d
    for line in open(PI):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("phase") != "decode" or r.get("tok_s") is None:
            continue
        key = "%.1fB %s" % (r["params_B"], r["tag"])
        d[key][r["threads"]].append(r["tok_s"])
    return d


def draw_laptop(ax, h, title, roof=None, legend=True):
    ts = sorted(h)
    med = [st.median(h[t]) for t in ts]
    p25 = [pct(h[t], 25) for t in ts]
    p75 = [pct(h[t], 75) for t in ts]
    lo = [min(h[t]) for t in ts]
    ax.fill_between(ts, p25, p75, alpha=0.28, linewidth=0,
                    label="interquartile range")
    ax.plot(ts, med, marker="o", markersize=3.0, linewidth=1.4, label="median")
    ax.plot(ts, lo, marker="v", markersize=2.6, linewidth=0.9, linestyle=":",
            label="worst sample")
    ax.axvline(PCORES, color="0.5", linestyle="--", linewidth=0.9)
    peak_t = max(h, key=lambda t: st.median(h[t]))
    ax.axvline(peak_t, color="0.5", linestyle="-.", linewidth=0.9)
    span = max(p75) - min(lo)
    ax.text(PCORES + 0.35, min(lo) + 0.03 * span, "6 P-cores", rotation=90,
            va="bottom", ha="left", fontsize=6.5, color="0.35")
    ax.text(peak_t + 0.35, min(lo) + 0.03 * span, "peak", rotation=90,
            va="bottom", ha="left", fontsize=6.5, color="0.35")
    if roof is not None:
        ax.axhline(roof, color="0.25", linestyle="-", linewidth=0.8)
        ax.set_ylim(top=roof + 0.16 * span)
        ax.text(max(ts), roof, "%.1f GB/s DRAM ceiling " % BW_CEILING_GBS,
                fontsize=6.0, va="bottom", ha="right", color="0.25")
    ax.set_xlabel("threads")
    ax.set_ylabel("decode tok/s")
    ax.set_title(title, fontsize=8)
    if legend:
        ax.legend(fontsize=6.0, loc="lower right", frameon=False)
    ax.grid(alpha=0.3, linewidth=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42        # never Type-3
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.size"] = 8

    lap, nbytes, dropped = load_laptop()
    pi = load_pi()
    if "qwen0.5b" not in lap:
        raise SystemExit("no randomised laptop data; run the sweep first")
    if "qwen7b" not in lap:
        raise SystemExit("no randomised 7B data; expected %s" % SEVENB)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(7.1, 2.45), gridspec_kw={"width_ratios": [1.3, 1.3, 1.0]})

    draw_laptop(ax1, lap["qwen0.5b"], "Qwen 0.5B (0.49 GB), randomised passes")
    draw_laptop(ax2, lap["qwen7b"], "Qwen 7.6B (4.68 GB), same protocol",
                roof=BW_CEILING_GBS * 1e9 / nbytes["qwen7b"], legend=False)

    # ---- right: the Pi, to show the band nearly vanish on homogeneous cores --
    for name, mk in zip(sorted(pi)[:3], ["o", "s", "^"]):
        v = pi[name]
        pts = sorted(v)
        m = [st.median(v[t]) for t in pts]
        peak = max(m)
        ax3.plot(pts, [100 * x / peak for x in m], marker=mk, markersize=3.0,
                 linewidth=1.2, label=name)
    ax3.set_xlabel("threads")
    ax3.set_ylabel("% of peak")
    ax3.set_title("Pi 5 (4 identical A76)", fontsize=8)
    ax3.set_xticks([1, 2, 3, 4])
    ax3.legend(fontsize=5.5, loc="lower left", frameon=False)
    ax3.grid(alpha=0.3, linewidth=0.5)

    fig.tight_layout(pad=0.6)
    fig.savefig(args.out)
    print("wrote", args.out)
    for path, n in dropped.items():
        print("  %-30s stall-excluded cells: %d" % (path, n))
    for tag in ("qwen0.5b", "qwen7b"):
        h = lap[tag]
        pt = max(h, key=lambda t: st.median(h[t]))
        tp = max(h)
        print("  %-9s peak %.2f tok/s at t=%d; at t=%d median %.2f, worst %.2f"
              % (tag, st.median(h[pt]), pt, tp, st.median(h[tp]), min(h[tp])))
    print("  models available: laptop %d, pi %d" % (len(lap), len(pi)))


if __name__ == "__main__":
    main()
