# What is in this directory, which files the paper uses, and which it does not

This paper is partly about measurement protocol, so the release has to be
explicit about which of its own files are trustworthy. Three of them are not,
and they are kept deliberately, because they are the evidence for the paper's
methodological claim.

## Use these

| file | rows | what it is |
|---|---|---|
| `randomized_sweep.jsonl` | 150 | The laptop thread curve for Qwen 0.5B, 5 passes in different seeded permutations, one invocation per cell with cooldowns. **This is the only laptop 0.5B thread-curve data the paper draws on.** |
| `randomized_sweep_sizes.jsonl` | 450 | The same protocol on Llama 1.2B, Qwen 1.5B and Qwen 3.1B. 75 cells each, prefill and decode rows per cell. |
| `randomized_sweep_7b.jsonl` | 150 | The same protocol on Qwen2.5-7B-Instruct q4\_k\_m, 4,683,073,632 bytes over two shards. 75 cells, 68 of which survive the stall exclusion below. The large end of the size contrast. |
| `order_control.jsonl` | 66 | Ascending and descending passes in one laptop session, bracketed by three reference cells. The evidence that sequential sweeps here are confounded. |
| `pi_order_control.jsonl` | 22 | The same control on the Raspberry Pi, with temperature and throttle flags. The evidence that the Pi is not. |
| `mask_validation.jsonl` | 10 | Proof that `--cpu-mask` actually binds on this build: eight threads confined to one logical processor run at 0.16 of eight free threads on decode. |
| `homo_sweep.jsonl`, `homo_sweep_add*.jsonl` | 168 | Pi thread curves, 10 repetitions, targets alternating small and large so file size is decorrelated from elapsed time. |
| `homo_perf*.jsonl` | 21 | Differential `perf` counters, $(I_{256}-I_{128})/\Delta$tokens, on the same sha256-verified artifacts as the curves. Three model families, two architectures. |
| `pi_decode_control.jsonl` | 12 | The original three-model Pi control. |
| `headroom_test.jsonl` | 32 | Four arms at 10 repetitions each, order permuted per pass, testing whether occupying the spare logical processors reproduces the oversubscribed tail. A = 16 threads on processors 0-15 with nothing else running, B = the same with four spinners pinned to 16-19, C = 20 threads on all twenty, D = A repeated as a drift check. |

## The stall exclusion, and why it is a rule rather than a judgement call

The 7B sweep ran for nine hours and the machine stalled during it. Seven cells
are contaminated. The rule that removes them is:

> **Exclude a cell whose wall-clock duration exceeds twice the median duration
> of the cells at its own thread count.**

Duration is the gap between consecutive `read_at` timestamps once cells are
sorted by `pass` then `position_in_pass`, so it includes the fixed 20 s cooldown.
The first cell of a run has no predecessor, has no derivable duration, and is
kept.

The per-thread-count reference is the whole rule, not a refinement. Decode at
one thread is slow, so a single-thread cell legitimately takes about 230 s where
the rest of the grid takes 70 to 124 s. Any fixed threshold gets it wrong in
both directions at once: set it high enough to keep clean single-thread cells
and it keeps a contaminated one, set it low enough to catch that and it discards
clean cells everywhere else.

What it does to `randomized_sweep_7b.jsonl`:

| pass,pos | threads | duration | threshold | cell median tok/s | worst sample |
|---|---|---|---|---|---|
| 2,10 | 12 | 7641 s | 160 s | 9.34 | 8.365 |
| 2,11 | 3 | 620 s | 204 s | 7.62 | 0.238 |
| 2,12 | 4 | 2573 s | 184 s | 8.69 | 8.575 |
| 3,7 | 14 | 5238 s | 150 s | 10.56 | 6.427 |
| 3,8 | 10 | 9302 s | 152 s | 10.74 | 0.041 |
| 4,5 | 1 | 588 s | 464 s | 1.25 | 1.227 |
| 5,3 | 1 | 506 s | 464 s | 1.26 | 1.244 |

68 of 75 cells survive. Nine thread counts keep all five passes and five keep
four. **One thread count, t=1, keeps only three**, because both of its
contaminated cells also ran long enough to be caught. Nothing in the paper rests
on t=1 and the table reports it on that reduced basis.

Note that five of the seven excluded cells returned medians inside the normal
range. They are not detectable from throughput and would survive any outlier
test applied to it. They are detectable only because the protocol records
`read_at` and `position_in_pass` on every cell.

Applied unchanged to `randomized_sweep.jsonl` and `randomized_sweep_sizes.jsonl`
the rule excludes nothing.

The rule lives in `stall_excluded()` in `code/analyze_headroom_sizes.py` and is
imported by `code/gen_fig_threads.py` and `code/fill_laptop_rand.py`, so the
figure, the tables and the macros all describe the same set of cells.

## Do not use these, and here is why they are here

| file | rows | why it is not usable |
|---|---|---|
| `laptop_sweep_seq.jsonl` | 61 | Sequential ascending sweep, two models. Thread count is confounded with elapsed time. Kept because it is half of the pair that demonstrates the problem. |
| `laptop_sweep_rerun.jsonl` | 30 | The same configuration, same binary, same file, measured again. It disagrees with the above by 61% at twenty threads while agreeing to 2% at one. **That disagreement is the finding**, not a defect in either file. |
| `laptop_coreid_contaminated.jsonl` | 80 | A per-logical-processor scan run while the machine was doing other work, including builds and web requests. Pass agreement is $r=-0.15$ and one core reads 18.0 tok/s in one pass and 52.6 in the other. Kept as a worked example of what contamination looks like, and as a record that the author caused it. |
| `homo_sweep_v1_perturbed.jsonl` | 72 | Superseded Pi sweep whose power samplers leaked; 36 were still running hours later and the board's load average reached 32. Superseded by `homo_sweep.jsonl`. |

## Reproducing

```bash
python code/fill_laptop_rand.py     # laptop macros, randomised data only
python code/fill_mech.py            # the 21-artifact mechanism table
python code/fit_model.py            # the two-term model, 13 of 16
python code/fill_energy.py          # energy per token
python code/analyze_randomized.py   # the distribution, not the median
python code/analyze_headroom_sizes.py   # headroom test + 7B + the size contrast
python code/gen_fig_threads.py          # Figure 1, same exclusion as the tables
```

`analyze_headroom_sizes.py` recomputes every number in the headroom and
size-scaling sections from the files above and writes them to
`paper/numbers_new.tex` as macros, so nothing in those sections is typed by hand.

Check any sweep for the confound with the portfolio tool:

```bash
python ../tools/order_control.py data/order_control.jsonl \
    --value tok_s_median --sweep threads --where phase=decode   # exits 2
python ../tools/order_control.py data/pi_order_control.jsonl \
    --value tok_s_median --sweep threads --where phase=decode   # exits 0
```

## One number that is not recomputable here

The 53.9 GB/s DRAM ceiling for the i7-12700H was measured once with a separate
bandwidth probe and is carried as a constant. Everything else in the paper is
derived from files in this directory.
