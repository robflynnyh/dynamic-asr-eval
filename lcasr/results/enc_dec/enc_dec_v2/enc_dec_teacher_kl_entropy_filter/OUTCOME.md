# Encoder-decoder teacher-KL entropy-filter outcome

Generated from `aggregate.py` over the current pickles in this directory.
All rows have `n=1`. This folder does not contain matching no-adapt baseline
pickles, so the summary reports absolute WER rather than baseline deltas. The
saved rows use beam5/lp0.5 decoding; compare them only against a same-decode
baseline, not against greedy results from another folder.

## TEDLIUM

| Rank | Setting | WER | Ins | Del | Sub |
|---:|---|---:|---:|---:|---:|
| 1 | `tedlium-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em7-freq6_width34_time0` | 10.28 | 1.77 | 3.37 | 5.14 |
| 2 | `tedlium-test-teacher_kl-beam5_lp0p5-epoch-1-lr-3em7-no_aug` | 10.33 | 1.44 | 3.62 | 5.28 |
| 3 | `tedlium-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em7-no_aug` | 10.33 | 1.44 | 3.62 | 5.28 |
| 4 | `tedlium-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em6-no_aug` | 10.34 | 1.44 | 3.61 | 5.29 |
| 5 | `tedlium-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em6-freq6_width34_time0` | 10.60 | 1.68 | 3.59 | 5.34 |

The low-confidence entropy filter keeps the TEDLIUM results stable but does
not improve the best saved teacher-KL result. The best row remains an
augmented `freq6_width34_time0` setting at `lr=1e-7`.

## Earnings22

| Rank | Setting | WER | Ins | Del | Sub |
|---:|---|---:|---:|---:|---:|
| 1 | `earnings22-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em6-no_aug` | 28.74 | 4.79 | 5.94 | 18.01 |
| 2 | `earnings22-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em7-no_aug` | 28.74 | 4.78 | 5.94 | 18.02 |
| 3 | `earnings22-test-teacher_kl-beam5_lp0p5-epoch-1-lr-3em7-no_aug` | 28.74 | 4.79 | 5.94 | 18.01 |
| 4 | `earnings22-test-teacher_kl-beam5_lp0p5-epoch-1-lr-3em7-freq6_width34_time0` | 29.39 | 5.87 | 5.22 | 18.29 |
| 5 | `earnings22-test-teacher_kl-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0` | 29.59 | 6.14 | 5.36 | 18.10 |

The entropy filter does not materially change the safest Earnings22 behavior:
`no_aug` remains the best family, while augmented settings still tend to raise
insertions and WER.

## Takeaway

The entropy/low-confidence filter is not harmful, but it does not clearly beat
the original teacher-KL sweep. It mainly confirms that the teacher-KL objective
is already relatively stable compared with hard teacher-CE.
