#!/usr/bin/env python3
"""Figure 1 for the Hybrid-Core Decode Cliff paper.
Left: decode tok/s vs thread count on the i7-12700H (6P+8E), normalized to each
model's peak, showing the cliff past the P-core boundary. Right: the Raspberry Pi 5
homogeneous-core control (no cliff). Type-42 fonts (no Type-3, per venue rules)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 9

# --- Laptop primary data (i7-12700H, Q4_K_M), decode tok/s ---
laptop = {
    "Qwen 0.5B":  {1: 21.7, 4: 69.4, 8: 90.2, 14: 86.8, 20: 56.4},
    "Llama 1.2B": {1: 12.7, 4: 38.5, 8: 49.7, 14: 48.8, 20: 40.7},
    "Qwen 1.5B":  {1: 10.7, 4: 33.1, 8: 39.1, 14: 38.7, 20: 32.1},
    "Qwen 3.1B":  {1: 5.6,  4: 17.0, 8: 20.6, 14: 20.5, 20: 18.4},
    "Qwen 7.6B":  {1: 2.3,  4: 7.8,  8: 9.6,  14: 10.4, 20: 10.1},
}
# --- Pi 5 control (4x Cortex-A76), decode tok/s: read from fresh sweep JSONL ---
NAME = {"qwen0.5b-q4km.gguf": "Qwen 0.5B", "llama1b-q4km.gguf": "Llama 1.2B",
        "qwen1.5b-q4km.gguf": "Qwen 1.5B"}
pi = {}
with open("../data/pi_decode_control.jsonl") as f:
    for line in f:
        r = json.loads(line)
        pi.setdefault(NAME[r["model"]], {})[r["threads"]] = r["decode_tps"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.5, 1]})

# Left panel: laptop, normalized to peak
markers = ["o", "s", "^", "D", "v"]
for (name, d), mk in zip(laptop.items(), markers):
    ts = sorted(d)
    peak = max(d.values())
    ax1.plot(ts, [100 * d[t] / peak for t in ts], marker=mk, markersize=4, label=name, linewidth=1.4)
ax1.axvline(6, color="0.5", linestyle="--", linewidth=1)
ax1.text(6.2, 62, "6 P-cores", rotation=90, va="bottom", ha="left", fontsize=7.5, color="0.35")
ax1.set_xlabel("threads")
ax1.set_ylabel("decode throughput (% of peak)")
ax1.set_title("i7-12700H (6 P + 8 E cores)", fontsize=9)
ax1.set_xticks([1, 4, 8, 14, 20])
ax1.set_ylim(55, 103)
ax1.legend(fontsize=6.8, loc="lower left", framealpha=0.9)
ax1.grid(True, alpha=0.25)

# Right panel: Pi control, normalized to peak
for (name, d), mk in zip(pi.items(), markers):
    ts = sorted(d)
    peak = max(d.values())
    ax2.plot(ts, [100 * d[t] / peak for t in ts], marker=mk, markersize=4, label=name, linewidth=1.4)
ax2.set_xlabel("threads")
ax2.set_title("Pi 5 (4 identical A76)", fontsize=9)
ax2.set_xticks([1, 2, 3, 4])
ax2.set_ylim(55, 103)
ax2.legend(fontsize=6.8, loc="lower right", framealpha=0.9)
ax2.grid(True, alpha=0.25)

fig.tight_layout(pad=0.4)
fig.savefig("fig_cliff.pdf", bbox_inches="tight")
print("wrote fig_cliff.pdf")
