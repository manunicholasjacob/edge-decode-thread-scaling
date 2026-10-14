#!/usr/bin/env python3
"""Recompute the README numbers that the seven macro scripts do not write.

The seven commands in the README regenerate paper/numbers_*.tex, and `git diff`
checks those. A handful of numbers in README.md and data/README.md come from
elsewhere: the order-control tool, a mask-validation file, and the
sequential sweeps kept as evidence of the ordering confound. This script
recomputes each one from data/ and fails if any no longer rounds to the value
the README states, so every number in the README has a check behind it.

It also checks the one laptop macro that fill_laptop_rand.py takes as a
constant (LapSeqDrift, the session drift on identical work) against what
tools/order_control.py measures from data/order_control.jsonl.

Standard library only. Exit status 0 if everything matches, 1 otherwise.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
TOOL = os.path.join(HERE, "tools", "order_control.py")

failures = []


def rows(name):
    with open(os.path.join(DATA, name)) as f:
        return [json.loads(line) for line in f if line.strip()]


def check(label, got, want, ndigits):
    ok = round(got, ndigits) == want
    print("%-58s %10.4f -> %s (README: %s)" % (label, got, "ok" if ok else "MISMATCH", want))
    if not ok:
        failures.append(label)


def fall_peak_to_20(cells):
    """Percent fall from the best decode median to the 20-thread median."""
    return 100.0 * (1.0 - cells[20] / max(cells.values()))


# 1. --cpu-mask binds: eight threads on one logical processor vs eight free.
decode = {r["label"]: r["tok_s_median"] for r in rows("mask_validation.jsonl")
          if r["phase"] == "decode"}
check("mask_validation: pin8 / free8 decode median", decode["pin8"] / decode["free8"], 0.16, 2)

# 2. The same 0.5B configuration falls by 15% to 61% depending on order alone.
falls = {}
for name in ("laptop_sweep_seq.jsonl", "laptop_sweep_rerun.jsonl", "order_control.jsonl"):
    curves = {}
    for r in rows(name):
        if r.get("tag") == "qwen0.5b" and r.get("phase") == "decode":
            arm = r.get("arm", "default")
            if arm.startswith("ref_"):
                continue
            curves.setdefault(arm, {})[r["threads"]] = r["tok_s_median"]
    for arm, cells in curves.items():
        if 20 in cells:
            falls["%s/%s" % (name, arm)] = fall_peak_to_20(cells)
for k in sorted(falls):
    print("   fall peak->20, %-44s %5.1f%%" % (k, falls[k]))
check("smallest order-dependent fall, percent", min(falls.values()), 15, 0)
check("largest order-dependent fall, percent", max(falls.values()), 61, 0)


# 3. The order-control tool's verdicts and the numbers quoted from them.
def run_tool(name):
    proc = subprocess.run(
        [sys.executable, TOOL, os.path.join(DATA, name), "--value", "tok_s_median",
         "--sweep", "threads", "--where", "phase=decode"],
        capture_output=True, text=True)
    return proc.returncode, proc.stdout


code, out = run_tool("order_control.jsonl")
if code != 2:
    failures.append("order_control.jsonl should be flagged (exit 2), got %d" % code)
drift = float(re.search(r"drifts ([0-9.]+)% on identical work", out).group(1))
direction = float(re.search(r"sweep direction changes a point by ([0-9.]+)%", out).group(1))
check("laptop session drift on identical work, percent", drift, 14.3, 1)
check("laptop change from sweep direction alone, percent", direction, 71.0, 1)

code, out = run_tool("pi_order_control.jsonl")
if code != 0:
    failures.append("pi_order_control.jsonl should pass (exit 0), got %d" % code)
pi = float(re.search(r"largest disagreement: [+-]?([0-9.]+)%", out).group(1))
check("Pi largest ascending/descending disagreement, percent", pi, 0.2, 1)

# 4. The macro fill_laptop_rand.py carries as a constant must equal the tool's drift.
with open(os.path.join(HERE, "paper", "numbers_laptop.tex")) as f:
    macro = re.search(r"\\newcommand\{\\npLapSeqDrift\}\{([0-9.]+)\}", f.read())
check("npLapSeqDrift macro vs tool drift", float(macro.group(1)), round(drift, 1), 1)

if failures:
    print("\nFAILED: " + "; ".join(failures))
    sys.exit(1)
print("\nAll README numbers outside the macro files recompute from data/.")
