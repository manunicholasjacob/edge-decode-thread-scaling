#!/usr/bin/env python3
"""Classify this machine's logical processors from measurement, and emit masks.

Phase B benchmarks one thread pinned to each logical processor, twice, in
opposite orders. This turns that into a P/E classification and the CPU masks
Phase C needs, so the placement experiment never hardcodes an enumeration.

Why not just assume logical 0-11 are the six hyperthreaded P-cores and 12-19 the
eight E-cores, which is the documented Alder Lake order? Because a pilot run on
this machine contradicted it: mask bit 18 gave the FASTEST prefill in its batch
and bit 12 one of the slowest. Either the runtime's mask bits are not OS logical
processor numbers, or the enumeration is not what the documentation implies. The
paper does not need to resolve which, and should not guess: it needs to know
which bits are fast and which are slow, and that is measurable.

CLASSIFICATION. Prefill, not decode, because prefill is compute bound and
separates the core types, while single-thread decode does not (a pilot found
23.3 against 24.0 tok/s on cores whose prefill differed by 26%). Cores are split
at the largest gap in the sorted per-core medians, which needs no threshold
chosen in advance. The split is reported with the gap size so a reader can see
whether it was clean.

DRIFT CHECK. A per-core property is stable between the ascending and descending
passes; thermal or turbo drift over a sequential scan reverses. The two passes
are compared and the correlation reported. A single anomalous core in one pass
(logical 8 in the pilot) shows up here as a pass disagreement rather than as a
core type.
"""
import argparse
import collections
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(HERE, "data", "laptop_coreid.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default=DEFAULT)
    ap.add_argument("--phase", default="prefill", choices=["prefill", "decode"])
    args = ap.parse_args()

    by_core = collections.defaultdict(dict)       # core -> pass -> value
    for line in open(args.src):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("phase") != args.phase or "cpu_mask" not in r:
            continue
        bit = int(r["cpu_mask"], 16).bit_length() - 1
        s = r.get("samples_ts") or []
        by_core[bit][r.get("arm", "?")] = st.median(s) if s else r["tok_s_mean"]

    if not by_core:
        raise SystemExit("no %s records in %s" % (args.phase, args.src))

    cores = sorted(by_core)
    med = {c: st.median(list(by_core[c].values())) for c in cores}

    # Pass agreement: a real core property survives reversing the scan order.
    ups = [by_core[c].get("coreid_up") for c in cores]
    downs = [by_core[c].get("coreid_down") for c in cores]
    paired = [(u, d) for u, d in zip(ups, downs) if u and d]
    if len(paired) > 2:
        mu = st.mean(u for u, _ in paired)
        md = st.mean(d for _, d in paired)
        num = sum((u - mu) * (d - md) for u, d in paired)
        den = (sum((u - mu) ** 2 for u, _ in paired) ** 0.5 *
               sum((d - md) ** 2 for _, d in paired) ** 0.5)
        print("pass agreement: r=%+.2f over %d cores measured twice"
              % (num / den if den else float("nan"), len(paired)))
        worst = max(paired, key=lambda p: abs(p[0] - p[1]))
        print("  largest single-core disagreement: %.1f vs %.1f tok/s"
              % worst)
    else:
        print("pass agreement: only %d cores measured twice, cannot check drift"
              % len(paired))

    # Split at the largest gap in the sorted values: no threshold to choose.
    order = sorted(cores, key=lambda c: med[c])
    vals = [med[c] for c in order]
    gaps = [(vals[i + 1] - vals[i], i) for i in range(len(vals) - 1)]
    gap, idx = max(gaps)
    slow = set(order[:idx + 1])
    fast = set(order[idx + 1:])
    within = max(max(vals[:idx + 1]) - min(vals[:idx + 1]),
                 max(vals[idx + 1:]) - min(vals[idx + 1:])) if len(vals) > 2 else 0

    print()
    for c in cores:
        print("  bit %2d  %s  %6.2f tok/s %s"
              % (c, "SLOW" if c in slow else "fast", med[c],
                 "  ".join("%s=%.1f" % (k.replace("coreid_", ""), v)
                           for k, v in sorted(by_core[c].items()))))
    print()
    print("split gap %.2f tok/s; widest spread inside a group %.2f tok/s%s"
          % (gap, within, "" if gap > within else "  <-- NOT a clean split"))
    print("slow group: %d cores %s" % (len(slow), sorted(slow)))
    print("fast group: %d cores %s" % (len(fast), sorted(fast)))
    print("penalty of the slow group: %.1f%%"
          % (100 * (1 - st.mean(med[c] for c in slow) / st.mean(med[c] for c in fast))))

    def mask(bits):
        m = 0
        for b in bits:
            m |= 1 << b
        return hex(m)

    print()
    print("MASK_FAST=%s   # %d cores" % (mask(fast), len(fast)))
    print("MASK_SLOW=%s   # %d cores" % (mask(slow), len(slow)))
    print("MASK_ALL=%s" % mask(cores))
    print("N_FAST=%d" % len(fast))
    print("N_SLOW=%d" % len(slow))


if __name__ == "__main__":
    main()
