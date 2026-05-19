# Encoder-decoder dynamic-eval teacher-CE outcome

Generated from `aggregate.py` over the current pickles in this directory.
All rows have `n=1`, so the results are single-run snapshots. The current
folder contains beam5/lp0.5 no-adapt baselines for both datasets, so deltas
below are relative to the matching beam decode baseline. These deltas should
not be compared against older greedy/no-adapt baselines; the final decode and
teacher pseudo-label decode are part of the experimental condition.

## Baselines

| Dataset | Setting | WER | Ins | Del | Sub |
|---|---|---:|---:|---:|---:|
| `tedlium` | `tedlium-test-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug` | 10.33 | 1.44 | 3.62 | 5.28 |
| `earnings22` | `earnings22-test-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug` | 28.72 | 4.79 | 5.94 | 18.00 |

## TEDLIUM

| Rank | Setting | WER | Delta | Ins | Del | Sub |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `tedlium-test-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0` | 10.08 | -0.26 | 1.45 | 3.66 | 4.97 |
| 2 | `tedlium-test-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq6_width34_time0` | 10.10 | -0.23 | 1.56 | 3.51 | 5.03 |
| 3 | `tedlium-test-teacher_ce-beam5_lp0p5-epoch-1-lr-3em7-freq6_width34_time0` | 10.15 | -0.18 | 1.41 | 3.70 | 5.04 |

Teacher-CE improves TEDLIUM modestly. The best setting is `lr=1e-7` with
`freq3_width24_time0`, improving WER by 0.26 absolute points versus the
matching no-adapt beam baseline.

## Earnings22

| Rank | Setting | WER | Delta | Ins | Del | Sub |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `earnings22-test-teacher_ce-beam5_lp0p5-epoch-1-lr-3em7-freq3_width24_time0` | 31.44 | +2.72 | 8.43 | 4.79 | 18.22 |
| 2 | `earnings22-test-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq6_width34_time0` | 31.45 | +2.72 | 7.83 | 4.81 | 18.80 |
| 3 | `earnings22-test-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0` | 31.88 | +3.15 | 9.14 | 4.22 | 18.51 |

Teacher-CE is not beneficial on Earnings22 in this sweep. Every adapted
setting is worse than the no-adapt beam baseline, and the `no_aug` variants
are severe failures driven by insertion and substitution growth.

## Takeaway

Teacher-forced CE with beam5/lp0.5 is promising on TEDLIUM but unsafe on
Earnings22. For a single robust setting, this outcome supports avoiding plain
teacher-CE adaptation on noisier data unless an additional reliability gate or
softer target objective is used.
