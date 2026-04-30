# Encoder-decoder teacher-KL outcome

Generated from `aggregate.py` over the current pickles in this directory.
All rows have `n=1`. This folder does not contain matching no-adapt baseline
pickles, so the summary reports absolute WER rather than baseline deltas. The
saved rows use beam5/lp0.5 decoding; compare them only against a same-decode
baseline, not against greedy results from another folder.

## TEDLIUM

| Rank | Setting | WER | Ins | Del | Sub |
|---:|---|---:|---:|---:|---:|
| 1 | `tedlium-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em7-freq6_width34_time0` | 10.24 | 1.60 | 3.46 | 5.18 |
| 2 | `tedlium-test-teacher_kl-beam5_lp0p5-epoch-1-lr-3em7-freq6_width34_time0` | 10.25 | 1.63 | 3.48 | 5.14 |
| 3 | `tedlium-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em7-no_aug` | 10.32 | 1.44 | 3.61 | 5.27 |
| 4 | `tedlium-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em6-no_aug` | 10.33 | 1.44 | 3.61 | 5.28 |
| 5 | `tedlium-test-teacher_kl-beam5_lp0p5-epoch-1-lr-3em7-freq3_width24_time0` | 10.36 | 1.59 | 3.62 | 5.15 |

The `1e-5` TEDLIUM follow-up settings are clear failures, with 23-26% WER and
large deletion rates. Excluding those, teacher-KL is stable around 10.2-11.3%
WER, with the best result from `lr=1e-7`, `freq6_width34_time0`.

## Earnings22

| Rank | Setting | WER | Ins | Del | Sub |
|---:|---|---:|---:|---:|---:|
| 1 | `earnings22-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em6-no_aug` | 28.73 | 4.78 | 5.94 | 18.01 |
| 2 | `earnings22-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em7-no_aug` | 28.73 | 4.79 | 5.94 | 18.00 |
| 3 | `earnings22-test-teacher_kl-beam5_lp0p5-epoch-1-lr-3em7-no_aug` | 28.75 | 4.80 | 5.94 | 18.01 |
| 4 | `earnings22-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0` | 29.15 | 6.71 | 4.58 | 17.86 |
| 5 | `earnings22-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em6-freq3_width24_time0` | 29.21 | 6.57 | 4.91 | 17.73 |

Teacher-KL is much safer than hard teacher-CE on Earnings22. The no-augmentation
rows are essentially flat around 28.7% WER, while augmented rows generally
increase insertions.

## Takeaway

Teacher-KL avoids the catastrophic Earnings22 insertion/substitution blow-ups
seen with hard teacher-CE, but it also provides only small gains on TEDLIUM.
The best overall settings are conservative: `no_aug` for Earnings22-like data,
and `freq6_width34_time0` with `lr=1e-7` or `3e-7` for TEDLIUM.
