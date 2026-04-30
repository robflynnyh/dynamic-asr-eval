# Encoder-decoder teacher-KL relaxed-filter outcome

Generated from `aggregate.py` over the current pickles in this directory.
All rows have `n=1`. This folder does not contain matching no-adapt baseline
pickles, so the summary reports absolute WER rather than baseline deltas. The
saved rows use beam5/lp0.5 decoding; compare them only against a same-decode
baseline, not against greedy results from another folder.

## TEDLIUM

| Rank | Setting | WER | Ins | Del | Sub |
|---:|---|---:|---:|---:|---:|
| 1 | `tedlium-test-teacher_kl-beam5_lp0p5-epoch-1-lr-3em7-freq6_width34_time0` | 10.29 | 1.62 | 3.58 | 5.09 |
| 2 | `tedlium-test-teacher_kl-beam5_lp0p5-epoch-1-lr-3em7-no_aug` | 10.32 | 1.44 | 3.62 | 5.27 |
| 3 | `tedlium-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em7-no_aug` | 10.33 | 1.44 | 3.62 | 5.28 |
| 4 | `tedlium-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em6-no_aug` | 10.34 | 1.44 | 3.62 | 5.28 |
| 5 | `tedlium-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em7-freq6_width34_time0` | 10.35 | 1.69 | 3.47 | 5.19 |

Relaxing the filters does not improve TEDLIUM over the best original teacher-KL
rows. The best result is still close to the no-augmentation baseline behavior,
with small differences among the top settings.

## Earnings22

| Rank | Setting | WER | Ins | Del | Sub |
|---:|---|---:|---:|---:|---:|
| 1 | `earnings22-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0` | 28.06 | 6.16 | 4.58 | 17.31 |
| 2 | `earnings22-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em6-no_aug` | 28.72 | 4.79 | 5.94 | 17.99 |
| 3 | `earnings22-test-teacher_kl-beam5_lp0p5-epoch-1-lr-3em7-no_aug` | 28.74 | 4.79 | 5.93 | 18.02 |
| 4 | `earnings22-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em7-no_aug` | 28.74 | 4.80 | 5.94 | 18.01 |
| 5 | `earnings22-test-teacher_kl-beam5_lp0p5-epoch-1-lr-3em7-freq3_width24_time0` | 28.96 | 6.77 | 4.72 | 17.47 |

Relaxed filters produce the best Earnings22 KL result in these folders:
`lr=1e-7`, `freq3_width24_time0` reaches 28.06% WER. This trades higher
insertions for lower deletion and substitution rates.

## Takeaway

The relaxed-filter variant is the most promising KL variant for Earnings22,
but not for TEDLIUM. If following up, center Earnings22 runs around `lr=1e-7`
with the lighter `freq3_width24_time0` augmentation and compare directly
against a no-adapt baseline in the same folder.
