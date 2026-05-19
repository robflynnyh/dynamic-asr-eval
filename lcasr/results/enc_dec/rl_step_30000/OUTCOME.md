# rl_step_30000

`rl_step_30000` is the 30K RL-trained checkpoint from the ROB-61/PR #11 lineage.

Checkpoint path: `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt`

Rows are one-epoch self-training snapshots. `Normal WER` is the matching
unadapted beam5/lp0.5 decode for the same checkpoint when available.

## Best Rows

| Dataset | Split | Mode | LR | Augmentation | Normal WER | Adapted WER | Adapted vs normal |
|---|---|---|---:|---|---:|---:|---:|
| chime6 | dev | teacher_ce | 1em8 | freq9_width44_time0 | 0.81157 | 0.99413 | +0.18255 |
| chime6 | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.85160 | 1.00000 | +0.14840 |
| earnings22 | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.23007 | 0.21365 | -0.01642 |
| rev16 | test | teacher_ce | 3em8 | freq9_width44_time0 | 0.17206 | 0.17176 | -0.00030 |
| tedlium | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.08386 | 0.07992 | -0.00393 |

## Full Checkpoint Table

| Dataset | Split | Mode | LR | Augmentation | Adapted WER | Normal WER | Delta vs normal | Relative vs normal | Ins | Del | Sub |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| chime6 | dev | teacher_ce | 1em8 | freq1_width12_time0 | 1.00000 | 0.81157 | +0.18843 | +23.22% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | freq3_width24_time0 | 1.00000 | 0.81157 | +0.18843 | +23.22% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | freq9_width44_time0 | 0.99413 | 0.81157 | +0.18255 | +22.49% | 0.00002 | 0.99329 | 0.00082 |
| chime6 | dev | teacher_ce | 1em8 | freq9_width44_time0_basic_repeat_filter | 0.99648 | 0.81157 | +0.18491 | +22.78% | 0.00003 | 0.99617 | 0.00028 |
| chime6 | dev | teacher_ce | 1em8 | no_aug | 1.00000 | 0.81157 | +0.18843 | +23.22% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq1_width12_time0 | 1.00000 | 0.81157 | +0.18843 | +23.22% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq3_width24_time0 | 1.00000 | 0.81157 | +0.18843 | +23.22% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq9_width44_time0 | 0.99869 | 0.81157 | +0.18712 | +23.06% | 0.00000 | 0.99863 | 0.00007 |
| chime6 | dev | teacher_ce | 3em8 | freq9_width44_time0_basic_repeat_filter | 0.99702 | 0.81157 | +0.18545 | +22.85% | 0.00000 | 0.99643 | 0.00059 |
| chime6 | dev | teacher_ce | 3em8 | no_aug | 1.00000 | 0.81157 | +0.18843 | +23.22% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em9 | freq3_width24_time0 | 1.45427 | 0.81157 | +0.64269 | +79.19% | 0.49376 | 0.62108 | 0.33942 |
| chime6 | test | teacher_ce | 1em7 | freq3_width24_time0 | 1.00000 | 0.85160 | +0.14840 | +17.43% | 0.00000 | 1.00000 | 0.00000 |
| earnings22 | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.21365 | 0.23007 | -0.01642 | -7.14% | 0.04152 | 0.04213 | 0.13000 |
| earnings22 | test | teacher_ce | 1em7 | freq6_width34_time0 | 0.23150 | 0.23007 | +0.00143 | +0.62% | 0.04644 | 0.04450 | 0.14056 |
| earnings22 | test | teacher_ce | 3em7 | freq3_width24_time0 | 0.21800 | 0.23007 | -0.01207 | -5.25% | 0.04434 | 0.04242 | 0.13124 |
| earnings22 | test | teacher_ce | 3em7 | freq6_width34_time0 | 0.23089 | 0.23007 | +0.00082 | +0.36% | 0.04802 | 0.04197 | 0.14090 |
| earnings22 | test | teacher_ce | 3em8 | freq9_width44_time0 | 0.24741 | 0.23007 | +0.01734 | +7.54% | 0.04957 | 0.04411 | 0.15373 |
| earnings22 | test | teacher_kl | 1em7 | freq3_width24_time0 | 0.21855 | 0.23007 | -0.01152 | -5.01% | 0.04293 | 0.04207 | 0.13355 |
| earnings22 | test | teacher_kl | 1em7 | freq6_width34_time0 | 0.23070 | 0.23007 | +0.00063 | +0.28% | 0.04638 | 0.04160 | 0.14272 |
| earnings22 | test | teacher_kl | 3em7 | freq3_width24_time0 | 0.21982 | 0.23007 | -0.01025 | -4.46% | 0.04271 | 0.04391 | 0.13320 |
| earnings22 | test | teacher_kl | 3em7 | freq6_width34_time0 | 0.23164 | 0.23007 | +0.00157 | +0.68% | 0.04677 | 0.04242 | 0.14245 |
| rev16 | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.23599 | 0.17206 | +0.06393 | +37.15% | 0.02280 | 0.13812 | 0.07507 |
| rev16 | test | teacher_ce | 1em7 | freq9_width44_time0 | 0.23628 | 0.17206 | +0.06421 | +37.32% | 0.02091 | 0.14114 | 0.07423 |
| rev16 | test | teacher_ce | 1em8 | freq1_width12_time0 | 0.48730 | 0.17206 | +0.31523 | +183.21% | 0.18942 | 0.05230 | 0.24558 |
| rev16 | test | teacher_ce | 1em8 | freq3_width24_time0 | 0.17357 | 0.17206 | +0.00151 | +0.88% | 0.03404 | 0.05714 | 0.08239 |
| rev16 | test | teacher_ce | 1em8 | freq9_width44_time0 | 0.17252 | 0.17206 | +0.00045 | +0.26% | 0.02982 | 0.06022 | 0.08248 |
| rev16 | test | teacher_ce | 1em8 | freq9_width44_time0_basic_repeat_filter | 0.17178 | 0.17206 | -0.00029 | -0.17% | 0.03006 | 0.06017 | 0.08155 |
| rev16 | test | teacher_ce | 1em8 | no_aug | 0.40760 | 0.17206 | +0.23554 | +136.89% | 0.01246 | 0.34156 | 0.05358 |
| rev16 | test | teacher_ce | 3em8 | freq1_width12_time0 | 0.19263 | 0.17206 | +0.02057 | +11.95% | 0.03622 | 0.06471 | 0.09171 |
| rev16 | test | teacher_ce | 3em8 | freq3_width24_time0 | 0.31086 | 0.17206 | +0.13879 | +80.66% | 0.01956 | 0.22675 | 0.06455 |
| rev16 | test | teacher_ce | 3em8 | freq9_width44_time0 | 0.17176 | 0.17206 | -0.00030 | -0.18% | 0.02974 | 0.05906 | 0.08296 |
| rev16 | test | teacher_ce | 3em8 | freq9_width44_time0_basic_repeat_filter | 0.17204 | 0.17206 | -0.00003 | -0.01% | 0.02905 | 0.06120 | 0.08179 |
| rev16 | test | teacher_ce | 3em8 | no_aug | 0.39313 | 0.17206 | +0.22107 | +128.48% | 0.01093 | 0.33274 | 0.04947 |
| rev16 | test | teacher_ce | 3em9 | freq3_width24_time0 | 0.39903 | 0.17206 | +0.22697 | +131.91% | 0.07600 | 0.18076 | 0.14227 |
| tedlium | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.07992 | 0.08386 | -0.00393 | -4.69% | 0.01067 | 0.02764 | 0.04161 |
| tedlium | test | teacher_ce | 1em7 | freq6_width34_time0 | 0.08286 | 0.08386 | -0.00099 | -1.18% | 0.01205 | 0.02867 | 0.04214 |
| tedlium | test | teacher_ce | 3em7 | freq3_width24_time0 | 0.08173 | 0.08386 | -0.00213 | -2.54% | 0.01042 | 0.02878 | 0.04253 |
| tedlium | test | teacher_ce | 3em7 | freq6_width34_time0 | 0.08414 | 0.08386 | +0.00028 | +0.34% | 0.01106 | 0.03094 | 0.04214 |
| tedlium | test | teacher_kl | 1em7 | freq3_width24_time0 | 0.08389 | 0.08386 | +0.00004 | +0.04% | 0.01155 | 0.02956 | 0.04278 |
| tedlium | test | teacher_kl | 1em7 | freq6_width34_time0 | 0.08350 | 0.08386 | -0.00035 | -0.42% | 0.01163 | 0.02949 | 0.04239 |
| tedlium | test | teacher_kl | 3em7 | freq3_width24_time0 | 0.08542 | 0.08386 | +0.00156 | +1.86% | 0.01389 | 0.02874 | 0.04278 |
| tedlium | test | teacher_kl | 3em7 | freq6_width34_time0 | 0.08379 | 0.08386 | -0.00007 | -0.08% | 0.01088 | 0.03020 | 0.04271 |
