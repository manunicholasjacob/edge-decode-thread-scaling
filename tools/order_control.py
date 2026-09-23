#!/usr/bin/env python3
"""Is a swept measurement confounded with elapsed time? Check before believing it.

WHY THIS IS A TOOL AND NOT A SCRIPT IN ONE PAPER. Paper 17 reported a 35.1%
decode cliff. The sweep that produced it walked thread counts in ascending order
inside a single benchmark invocation, so the highest thread counts were always
measured last, on the warmest and busiest machine. Re-measuring the same
configuration five times gave falls of 15.2, 35.1, 49.8, 55.4 and 61.4 percent
depending only on when each cell happened to run. The headline was a difference
between the beginning and the end of a sweep.

Four other papers in this portfolio sweep a parameter sequentially: thread counts
in Papers 12 and 16, temperature caps in Papers 1 and 3. They turn out to be
safe, but only because they run on a platform that happens to be stable, which
nobody had checked. This makes the check routine.

TWO THINGS TO MEASURE, and they answer different questions.

  DRIFT. Run identical work at the start, middle and end of a session. If those
  disagree, the platform changes under you and every sequential measurement on
  it is suspect. On a Raspberry Pi 5 this gives 0.5%. On an Intel laptop under
  Windows it gave 14.3%, and it drifted UPWARD, so it was background load
  draining away rather than heat, which means a cooldown would not have fixed it.

  ORDER. Run the sweep ascending, then descending. A real property of the swept
  variable is the same in both directions. Drift reverses. The Pi agreed to 0.2%
  at every point; the laptop differed by up to 71%.

Repetition does not substitute for either. Repeating a cell back to back
measures noise at one moment in the session, and is blind to a bias that is
constant within a cell and varies between them. Paper 17 had fifteen repetitions
at the cell that mattered and still got the wrong answer.

USAGE

    python tools/order_control.py results.jsonl --value tok_s --sweep threads

    # with an explicit reference cell and a session-position field
    python tools/order_control.py results.jsonl --value tok_s --sweep threads \\
        --position position_in_pass --arm arm

Expects JSON Lines with at least the sweep variable and the measured value.
Optional fields make the check sharper: an arm label (so ascending and
descending passes can be compared), and a position-within-session field (so
drift can be regressed directly).

Exit status is 0 if the data looks order-independent, 2 if it does not, so this
can gate a release.
"""
import argparse
import collections
import json
import math
import statistics as st
import sys


def pearson(x, y):
    if len(x) < 3:
        return float("nan")
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if not sx or not sy:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def load(path, value, sweep, arm, position):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" in r or value not in r or sweep not in r:
            continue
        if r[value] is None:
            continue
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path")
    ap.add_argument("--value", required=True,
                    help="field holding the measurement, e.g. tok_s")
    ap.add_argument("--sweep", required=True,
                    help="field holding the swept variable, e.g. threads")
    ap.add_argument("--arm", default="arm",
                    help="field labelling the pass; ascending/descending arms "
                         "are compared when present")
    ap.add_argument("--position", default="position_in_pass",
                    help="field holding position within the session")
    ap.add_argument("--where", action="append", default=[], metavar="FIELD=VALUE",
                    help="keep only rows where FIELD equals VALUE; repeatable. "
                         "Without it a file holding several phases silently "
                         "mixes them, which is how a check like this gives a "
                         "confident answer about nothing in particular.")
    ap.add_argument("--drift-limit", type=float, default=5.0,
                    help="percent; above this the platform is called unstable")
    args = ap.parse_args()

    rows = load(args.path, args.value, args.sweep, args.arm, args.position)
    for clause in args.where:
        if "=" not in clause:
            sys.exit("--where needs FIELD=VALUE, got %r" % clause)
        f, v = clause.split("=", 1)
        before = len(rows)
        rows = [r for r in rows if str(r.get(f)) == v]
        print("filter %s=%s: %d of %d rows" % (f, v, len(rows), before))
    if not rows:
        sys.exit("no usable rows in %s (need fields %r and %r)"
                 % (args.path, args.value, args.sweep))
    print("%d measurements from %s" % (len(rows), args.path))
    problems = []

    # --- reference cells, if any arm is named like a reference --------------
    refs = collections.defaultdict(list)
    for r in rows:
        a = str(r.get(args.arm, ""))
        if "ref" in a.lower():
            refs[a].append(r[args.value])
    if len(refs) > 1:
        meds = {a: st.median(v) for a, v in refs.items()}
        lo, hi = min(meds.values()), max(meds.values())
        drift = 100 * (1 - lo / hi)
        print("\nreference cells (identical work at different session positions):")
        for a in sorted(meds):
            print("   %-14s %.3f" % (a, meds[a]))
        print("   drift across session: %.1f%%" % drift)
        if drift > args.drift_limit:
            problems.append("platform drifts %.1f%% on identical work" % drift)
    else:
        print("\nno reference cells found (arm field %r); "
              "drift cannot be measured directly" % args.arm)

    # --- ascending vs descending -------------------------------------------
    arms = collections.defaultdict(dict)
    for r in rows:
        a = str(r.get(args.arm, "")).lower()
        if "asc" in a or "desc" in a:
            arms[a].setdefault(r[args.sweep], []).append(r[args.value])
    asc = next((v for k, v in arms.items() if "asc" in k), None)
    desc = next((v for k, v in arms.items() if "desc" in k), None)
    if asc and desc:
        common = sorted(set(asc) & set(desc))
        print("\nascending vs descending, %d shared points:" % len(common))
        worst, worst_at = 0.0, None
        for k in common:
            a, d = st.median(asc[k]), st.median(desc[k])
            diff = 100 * (d / a - 1)
            if abs(diff) > abs(worst):
                worst, worst_at = diff, k
            print("   %-10s %10.3f %10.3f  %+7.1f%%" % (k, a, d, diff))
        print("   largest disagreement: %+.1f%% at %s" % (worst, worst_at))
        if abs(worst) > args.drift_limit * 2:
            problems.append("sweep direction changes a point by %.1f%%" % abs(worst))
    else:
        print("\nno ascending/descending arms found; order effect not testable")

    # --- position regression, normalised within the swept variable ---------
    if any(args.position in r for r in rows):
        med = collections.defaultdict(list)
        for r in rows:
            med[r[args.sweep]].append(r[args.value])
        med = {k: st.median(v) for k, v in med.items()}
        res = collections.defaultdict(list)
        for r in rows:
            if args.position in r and med.get(r[args.sweep]):
                res[r[args.position]].append(r[args.value] / med[r[args.sweep]])
        xs = sorted(res)
        if len(xs) > 2:
            ys = [st.median(res[p]) for p in xs]
            fx = [float(x) for x in xs]
            mx, my = st.mean(fx), st.mean(ys)
            den = sum((x - mx) ** 2 for x in fx)
            slope = sum((x - mx) * (y - my) for x, y in zip(fx, ys)) / den if den else 0
            span = 100 * slope * (max(fx) - min(fx))
            print("\nposition effect (each sample normalised by its own %s median):"
                  % args.sweep)
            print("   slope %+.5f per position, %+.1f%% across %d positions"
                  % (slope, span, len(xs)))
            print("   correlation r=%+.2f" % pearson(fx, ys))
            if abs(span) > args.drift_limit:
                problems.append("position predicts %.1f%% of the measurement" % abs(span))

    print()
    if problems:
        print("ORDER-CONFOUNDED. Do not report a difference between the first and")
        print("last cells of this sweep as a property of the swept variable:")
        for p in problems:
            print("   - " + p)
        print("\nRemedy: several passes, each visiting the swept values in a")
        print("different permutation, one invocation per cell. See")
        print("paper17-decode-cliff/code/randomized_sweep.sh.")
        sys.exit(2)
    print("LOOKS ORDER-INDEPENDENT on the checks this data supports.")
    sys.exit(0)


if __name__ == "__main__":
    main()
