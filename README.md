# Decode thread scaling on hybrid and homogeneous edge CPUs

Measurement artifact for **"Use More Threads, Not Fewer: Thread Scaling for LLM
Decode on Hybrid and Homogeneous Edge CPUs"**, under review at IEEE
Transactions on Parallel and Distributed Systems.

Everything the paper claims about thread scaling is recomputed from the records
in `data/` by the scripts in `code/`. Nothing in the tables or figures is typed
by hand. This README states which numbers come out of this repository and which
one does not.

## What the paper found

Practitioner guidance for LLM decode on hybrid CPUs is to cap the thread count
at the performance-core count. On an Intel i7-12700H that advice costs 18.5% of
decode throughput: throughput keeps rising past the P-core boundary to a peak at
16 threads.

Past the peak the failure is not a throughput collapse. It is a collapse in
predictability. At 16 threads the coefficient of variation across samples is
2.7%; at 18 it is 29.4% and at 20 it is 31.0%, while the median only falls
20.8%. The upper quartile at 18 threads is still 95% of the peak's and the worst
run is 36% of it. Oversubscription costs reliability, not speed.

A controlled experiment names the cause. Holding the thread count at 16 and
occupying the four otherwise-spare logical processors reproduces the
oversubscribed spread (CV 31.2%) where leaving them free does not (CV 20.3%),
and the oversubscribed arm itself sits at 29.6%. The loaded arm is 1.6
percentage points from the oversubscribed arm and 10.9 from the free one.
Headroom is the mechanism.

All of this stayed invisible under a confound this work shares with the
literature: an ascending sweep ties thread count to elapsed time. The same
configuration on the same binary yielded falls between 15% and 61% depending
only on measurement order. Every laptop number the paper uses was therefore
remeasured under a randomized protocol: several passes, each visiting the thread
counts in a different seeded permutation, one process invocation per cell, with
a cooldown between cells and the pass and position recorded so drift can be
measured rather than assumed away.

## Reproducing

Python 3.11 or newer, standard library only, plus `matplotlib` for the figure.

```bash
python code/fill_laptop_rand.py       # laptop macros, randomized data only
python code/fill_mech.py              # the 21-artifact mechanism table
python code/fit_model.py              # the two-term model, 13 of 16
python code/fill_energy.py            # energy per token
python code/analyze_randomized.py     # the distribution, not the median
python code/analyze_headroom_sizes.py # headroom test, 7B sweep, size contrast
python code/gen_fig_threads.py        # Figure 1, same exclusion as the tables
```

The five `paper/numbers_*.tex` macro files these write are the ones the
manuscript includes, and they are committed here as generated. Running the
seven commands above and then `git diff` is therefore the check: it should
come back empty, which is what it does. Generated figure PDFs are not tracked,
because a PDF embeds its creation time and would differ on every build for
that reason alone.

To check any sweep for the ordering confound:

```bash
python tools/order_control.py data/order_control.jsonl \
    --value tok_s_median --sweep threads --where phase=decode   # flags the laptop
python tools/order_control.py data/pi_order_control.jsonl \
    --value tok_s_median --sweep threads --where phase=decode   # clears the Pi
```

The laptop file is flagged: 14.3% session drift on identical work and a 71.0%
change from sweep direction alone. The Pi file is not: the largest
pass-to-pass disagreement is 0.2%.

## What is in here

`data/README.md` is the authoritative guide to the records. It lists which files
the paper draws on, which three are deliberately retained because they are
*unusable* and that is the evidence for the methodological claim, and the
exclusion rule applied to the 7B sweep.

The short version:

- `randomized_sweep.jsonl`, `randomized_sweep_sizes.jsonl`,
  `randomized_sweep_7b.jsonl` are the laptop thread curves under the randomized
  protocol, from 0.5B to 7.6B.
- `headroom_test.jsonl` is the four-arm controlled experiment.
- `order_control.jsonl` and `pi_order_control.jsonl` are the evidence that the
  laptop is order-confounded and the Pi is not.
- `mask_validation.jsonl` shows `--cpu-mask` actually binds on this build:
  eight threads confined to one logical processor run at 0.16 of eight free
  threads.
- `homo_sweep*.jsonl` and `homo_perf*.jsonl` are the Raspberry Pi 5 curves and
  the differential `perf` counters on sha256-matched artifacts.
- `laptop_sweep_seq.jsonl`, `laptop_sweep_rerun.jsonl`,
  `laptop_coreid_contaminated.jsonl` and `homo_sweep_v1_perturbed.jsonl` are
  kept and are **not** to be used for results. Each one is documented in
  `data/README.md` with the reason.

### The 7B exclusion rule

The 7B sweep ran for nine hours and the machine stalled during it. Contaminated
cells are removed by a rule, not a judgement call: exclude a cell whose
wall-clock duration exceeds twice the median duration of the cells at its own
thread count. 68 of 75 cells survive. The seven excluded cells are listed
individually in `data/README.md`, and five of them returned medians inside the
normal range, which is why a median-based filter would not have caught them.

## What is not recomputable from this repository

- **The 53.9 GB/s DRAM ceiling for the i7-12700H.** It was measured once with a
  separate bandwidth probe and is carried through the paper as a constant.
  Everything else is derived from files in `data/`.
- **The model files.** GGUF artifacts are not redistributed. The sweep scripts
  download them from Hugging Face at run time and delete them afterwards; the
  byte count and, where the script records it, the sha256 of each artifact are
  stored in the measurement records, so the identity of what was measured is
  fixed even though the bytes are not shipped here.

## Hardware

- Intel Core i7-12700H, 6 performance cores and 8 efficiency cores, 20 logical
  processors, Windows.
- Raspberry Pi 5, Cortex-A76, 4 cores.

Inference is llama.cpp via `llama-bench`. The build commit is recorded in the
`build` field of every measurement row.

## License

MIT. See `LICENSE`.
