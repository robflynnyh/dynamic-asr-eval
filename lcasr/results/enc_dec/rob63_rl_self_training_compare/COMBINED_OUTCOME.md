# ROB-63 RL self-training comparison

Snapshot generated from completed ROB-63 pickles.
RL deltas compare each RL `step_30000` self-training cell against the matching old-seed cell.
Normal/unadapted deltas compare each self-training cell against the same checkpoint's normal decoding WER.

## Summary

- Completed 64 one-epoch result rows covering 4 datasets, 2 splits, 2 checkpoints, 2 training modes, 5 learning rates, 4 frequency-mask settings.
- The sweep used no teacher filtering. `Delta vs old seed` is only meaningful for the RL rows because old-seed rows are the matching-cell reference.
- chime6/dev: RL `step_30000` beats the matching old-seed cell in 0/7 cells; RL-vs-old absolute WER deltas span +0.00000 to +0.45427. Best RL cell is rl_step_30000 teacher_ce lr=1em8 freq1_width12_time0 at 1.00000 WER; best old-seed cell is old_seed teacher_ce lr=1em8 freq1_width12_time0 at 1.00000 WER. Relative change vs normal decoding spans +23.22% to +79.19% for RL and +19.85% to +19.85% for old seed.
- chime6/test: RL `step_30000` beats the matching old-seed cell in 0/1 cells; RL-vs-old absolute WER deltas span +0.00000 to +0.00000. Best RL cell is rl_step_30000 teacher_ce lr=1em7 freq3_width24_time0 at 1.00000 WER; best old-seed cell is old_seed teacher_ce lr=1em7 freq3_width24_time0 at 1.00000 WER. Relative change vs normal decoding spans +17.43% to +17.43% for RL and +15.64% to +15.64% for old seed.
- earnings22/test: RL `step_30000` beats the matching old-seed cell in 6/8 cells; RL-vs-old absolute WER deltas span -0.01454 to +0.00135. Best RL cell is rl_step_30000 teacher_ce lr=1em7 freq3_width24_time0 at 0.21365 WER; best old-seed cell is old_seed teacher_ce lr=3em7 freq3_width24_time0 at 0.21686 WER. Relative change vs normal decoding spans -7.14% to +0.68% for RL and -13.85% to -2.26% for old seed.
- rev16/test: RL `step_30000` beats the matching old-seed cell in 3/8 cells; RL-vs-old absolute WER deltas span -0.06387 to +0.30307. Best RL cell is rl_step_30000 teacher_ce lr=1em8 freq3_width24_time0 at 0.17357 WER; best old-seed cell is old_seed teacher_ce lr=3em8 freq3_width24_time0 at 0.17915 WER. Relative change vs normal decoding spans +0.88% to +183.21% for RL and +1.43% to +109.77% for old seed.
- tedlium/test: RL `step_30000` beats the matching old-seed cell in 8/8 cells; RL-vs-old absolute WER deltas span -0.13418 to -0.00457. Best RL cell is rl_step_30000 teacher_ce lr=1em7 freq3_width24_time0 at 0.07992 WER; best old-seed cell is old_seed teacher_ce lr=3em7 freq3_width24_time0 at 0.08747 WER. Relative change vs normal decoding spans -4.69% to +1.86% for RL and -1.67% to +143.98% for old seed.
- Sanity note: rev16 rl_step_30000 teacher_ce lr=1em8 freq1_width12_time0 is an outlier versus normal decoding (+183.21%, WER 0.48730); the error mix is dominated by deletion rate 0.05230.
- Sanity note: tedlium old_seed teacher_ce lr=1em7 freq6_width34_time0 is an outlier versus normal decoding (+143.98%, WER 0.21705); the error mix is dominated by deletion rate 0.16867.
- Sanity note: rev16 rl_step_30000 teacher_ce lr=1em8 no_aug is an outlier versus normal decoding (+136.89%, WER 0.40760); the error mix is dominated by deletion rate 0.34156.
- Sanity note: rev16 rl_step_30000 teacher_ce lr=3em9 freq3_width24_time0 is an outlier versus normal decoding (+131.91%, WER 0.39903); the error mix is dominated by deletion rate 0.18076.
- Sanity note: rev16 rl_step_30000 teacher_ce lr=3em8 no_aug is an outlier versus normal decoding (+128.48%, WER 0.39313); the error mix is dominated by deletion rate 0.33274.
- Sanity note: rev16 old_seed teacher_ce lr=3em9 freq3_width24_time0 is an outlier versus normal decoding (+109.77%, WER 0.37051); the error mix is dominated by deletion rate 0.14591.
- Sanity note: rev16 rl_step_30000 teacher_ce lr=3em8 freq3_width24_time0 is an outlier versus normal decoding (+80.66%, WER 0.31086); the error mix is dominated by deletion rate 0.22675.
- Sanity note: chime6 rl_step_30000 teacher_ce lr=3em9 freq3_width24_time0 is an outlier versus normal decoding (+79.19%, WER 1.45427); the error mix is dominated by deletion rate 0.62108.
- Sanity note: rev16 old_seed teacher_ce lr=1em8 no_aug is an outlier versus normal decoding (+45.73%, WER 0.25740); the error mix is dominated by deletion rate 0.16927.
- Sanity note: rev16 old_seed teacher_ce lr=3em8 no_aug is an outlier versus normal decoding (+40.42%, WER 0.24803); the error mix is dominated by deletion rate 0.15787.
- Sanity note: rev16 rl_step_30000 teacher_ce lr=1em7 freq3_width24_time0 is an outlier versus normal decoding (+37.15%, WER 0.23599); the error mix is dominated by deletion rate 0.13812.
- Sanity note: rev16 old_seed teacher_ce lr=3em8 freq1_width12_time0 is an outlier versus normal decoding (+36.92%, WER 0.24183); the error mix is dominated by deletion rate 0.14489.
- Sanity note: rev16 old_seed teacher_ce lr=1em7 freq3_width24_time0 is an outlier versus normal decoding (+34.69%, WER 0.23790); the error mix is dominated by deletion rate 0.14223.
- Sanity note: rev16 old_seed teacher_ce lr=1em8 freq3_width24_time0 is an outlier versus normal decoding (+34.43%, WER 0.23744); the error mix is dominated by deletion rate 0.14202.

## Unadapted vs adapted WER

This is the main readout. `Old normal WER` and `RL normal WER` are the unadapted beam5/lp0.5 decoding baselines for each checkpoint. The adapted columns are the one-epoch self-training WERs for the listed setting.

| Dataset | Split | Mode | LR | Augmentation | Old normal WER | Old adapted WER | Old adapted vs normal | RL normal WER | RL adapted WER | RL adapted vs normal | RL adapted vs old adapted |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| chime6 | dev | teacher_ce | 1em8 | freq1_width12_time0 | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 1.00000 | +0.18843 | +0.00000 |
| chime6 | dev | teacher_ce | 1em8 | freq3_width24_time0 | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 1.00000 | +0.18843 | +0.00000 |
| chime6 | dev | teacher_ce | 1em8 | no_aug | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 1.00000 | +0.18843 | +0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq1_width12_time0 | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 1.00000 | +0.18843 | +0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq3_width24_time0 | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 1.00000 | +0.18843 | +0.00000 |
| chime6 | dev | teacher_ce | 3em8 | no_aug | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 1.00000 | +0.18843 | +0.00000 |
| chime6 | dev | teacher_ce | 3em9 | freq3_width24_time0 | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 1.45427 | +0.64269 | +0.45427 |
| chime6 | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.86477 | 1.00000 | +0.13523 | 0.85160 | 1.00000 | +0.14840 | +0.00000 |
| earnings22 | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.25172 | 0.21806 | -0.03366 | 0.23007 | 0.21365 | -0.01642 | -0.00441 |
| earnings22 | test | teacher_ce | 1em7 | freq6_width34_time0 | 0.25172 | 0.24604 | -0.00568 | 0.23007 | 0.23150 | +0.00143 | -0.01454 |
| earnings22 | test | teacher_ce | 3em7 | freq3_width24_time0 | 0.25172 | 0.21686 | -0.03486 | 0.23007 | 0.21800 | -0.01207 | +0.00114 |
| earnings22 | test | teacher_ce | 3em7 | freq6_width34_time0 | 0.25172 | 0.23630 | -0.01542 | 0.23007 | 0.23089 | +0.00082 | -0.00541 |
| earnings22 | test | teacher_kl | 1em7 | freq3_width24_time0 | 0.25172 | 0.21720 | -0.03452 | 0.23007 | 0.21855 | -0.01152 | +0.00135 |
| earnings22 | test | teacher_kl | 1em7 | freq6_width34_time0 | 0.25172 | 0.23265 | -0.01908 | 0.23007 | 0.23070 | +0.00063 | -0.00194 |
| earnings22 | test | teacher_kl | 3em7 | freq3_width24_time0 | 0.25172 | 0.22223 | -0.02949 | 0.23007 | 0.21982 | -0.01025 | -0.00241 |
| earnings22 | test | teacher_kl | 3em7 | freq6_width34_time0 | 0.25172 | 0.23197 | -0.01975 | 0.23007 | 0.23164 | +0.00157 | -0.00033 |
| rev16 | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.17663 | 0.23790 | +0.06127 | 0.17206 | 0.23599 | +0.06393 | -0.00191 |
| rev16 | test | teacher_ce | 1em8 | freq1_width12_time0 | 0.17663 | 0.18423 | +0.00760 | 0.17206 | 0.48730 | +0.31523 | +0.30307 |
| rev16 | test | teacher_ce | 1em8 | freq3_width24_time0 | 0.17663 | 0.23744 | +0.06081 | 0.17206 | 0.17357 | +0.00151 | -0.06387 |
| rev16 | test | teacher_ce | 1em8 | no_aug | 0.17663 | 0.25740 | +0.08077 | 0.17206 | 0.40760 | +0.23554 | +0.15020 |
| rev16 | test | teacher_ce | 3em8 | freq1_width12_time0 | 0.17663 | 0.24183 | +0.06520 | 0.17206 | 0.19263 | +0.02057 | -0.04920 |
| rev16 | test | teacher_ce | 3em8 | freq3_width24_time0 | 0.17663 | 0.17915 | +0.00252 | 0.17206 | 0.31086 | +0.13879 | +0.13170 |
| rev16 | test | teacher_ce | 3em8 | no_aug | 0.17663 | 0.24803 | +0.07140 | 0.17206 | 0.39313 | +0.22107 | +0.14510 |
| rev16 | test | teacher_ce | 3em9 | freq3_width24_time0 | 0.17663 | 0.37051 | +0.19388 | 0.17206 | 0.39903 | +0.22697 | +0.02852 |
| tedlium | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.08896 | 0.08811 | -0.00085 | 0.08386 | 0.07992 | -0.00393 | -0.00819 |
| tedlium | test | teacher_ce | 1em7 | freq6_width34_time0 | 0.08896 | 0.21705 | +0.12809 | 0.08386 | 0.08286 | -0.00099 | -0.13418 |
| tedlium | test | teacher_ce | 3em7 | freq3_width24_time0 | 0.08896 | 0.08747 | -0.00149 | 0.08386 | 0.08173 | -0.00213 | -0.00574 |
| tedlium | test | teacher_ce | 3em7 | freq6_width34_time0 | 0.08896 | 0.08871 | -0.00025 | 0.08386 | 0.08414 | +0.00028 | -0.00457 |
| tedlium | test | teacher_kl | 1em7 | freq3_width24_time0 | 0.08896 | 0.09374 | +0.00478 | 0.08386 | 0.08389 | +0.00004 | -0.00985 |
| tedlium | test | teacher_kl | 1em7 | freq6_width34_time0 | 0.08896 | 0.09236 | +0.00340 | 0.08386 | 0.08350 | -0.00035 | -0.00886 |
| tedlium | test | teacher_kl | 3em7 | freq3_width24_time0 | 0.08896 | 0.09431 | +0.00535 | 0.08386 | 0.08542 | +0.00156 | -0.00890 |
| tedlium | test | teacher_kl | 3em7 | freq6_width34_time0 | 0.08896 | 0.09477 | +0.00581 | 0.08386 | 0.08379 | -0.00007 | -0.01099 |

## Full table

| Dataset | Split | Mode | LR | Augmentation | Checkpoint | Adapted WER | Unadapted WER | Delta vs unadapted | Relative vs unadapted | Delta vs old seed | Relative vs old seed | Ins | Del | Sub |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chime6 | dev | teacher_ce | 1em8 | freq1_width12_time0 | old_seed | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | freq1_width12_time0 | rl_step_30000 | 1.00000 | 0.81157 | +0.18843 | +23.22% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | freq3_width24_time0 | old_seed | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | freq3_width24_time0 | rl_step_30000 | 1.00000 | 0.81157 | +0.18843 | +23.22% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | no_aug | old_seed | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | no_aug | rl_step_30000 | 1.00000 | 0.81157 | +0.18843 | +23.22% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq1_width12_time0 | old_seed | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq1_width12_time0 | rl_step_30000 | 1.00000 | 0.81157 | +0.18843 | +23.22% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq3_width24_time0 | old_seed | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq3_width24_time0 | rl_step_30000 | 1.00000 | 0.81157 | +0.18843 | +23.22% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | no_aug | old_seed | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | no_aug | rl_step_30000 | 1.00000 | 0.81157 | +0.18843 | +23.22% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em9 | freq3_width24_time0 | old_seed | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em9 | freq3_width24_time0 | rl_step_30000 | 1.45427 | 0.81157 | +0.64269 | +79.19% | +0.45427 | +45.43% | 0.49376 | 0.62108 | 0.33942 |
| chime6 | test | teacher_ce | 1em7 | freq3_width24_time0 | old_seed | 1.00000 | 0.86477 | +0.13523 | +15.64% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | test | teacher_ce | 1em7 | freq3_width24_time0 | rl_step_30000 | 1.00000 | 0.85160 | +0.14840 | +17.43% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| earnings22 | test | teacher_ce | 1em7 | freq3_width24_time0 | old_seed | 0.21806 | 0.25172 | -0.03366 | -13.37% | +0.00000 | +0.00% | 0.03789 | 0.05292 | 0.12726 |
| earnings22 | test | teacher_ce | 1em7 | freq3_width24_time0 | rl_step_30000 | 0.21365 | 0.23007 | -0.01642 | -7.14% | -0.00441 | -2.02% | 0.04152 | 0.04213 | 0.13000 |
| earnings22 | test | teacher_ce | 1em7 | freq6_width34_time0 | old_seed | 0.24604 | 0.25172 | -0.00568 | -2.26% | +0.00000 | +0.00% | 0.05388 | 0.04881 | 0.14335 |
| earnings22 | test | teacher_ce | 1em7 | freq6_width34_time0 | rl_step_30000 | 0.23150 | 0.23007 | +0.00143 | +0.62% | -0.01454 | -5.91% | 0.04644 | 0.04450 | 0.14056 |
| earnings22 | test | teacher_ce | 3em7 | freq3_width24_time0 | old_seed | 0.21686 | 0.25172 | -0.03486 | -13.85% | +0.00000 | +0.00% | 0.04248 | 0.04902 | 0.12536 |
| earnings22 | test | teacher_ce | 3em7 | freq3_width24_time0 | rl_step_30000 | 0.21800 | 0.23007 | -0.01207 | -5.25% | +0.00114 | +0.53% | 0.04434 | 0.04242 | 0.13124 |
| earnings22 | test | teacher_ce | 3em7 | freq6_width34_time0 | old_seed | 0.23630 | 0.25172 | -0.01542 | -6.13% | +0.00000 | +0.00% | 0.04650 | 0.04934 | 0.14045 |
| earnings22 | test | teacher_ce | 3em7 | freq6_width34_time0 | rl_step_30000 | 0.23089 | 0.23007 | +0.00082 | +0.36% | -0.00541 | -2.29% | 0.04802 | 0.04197 | 0.14090 |
| earnings22 | test | teacher_kl | 1em7 | freq3_width24_time0 | old_seed | 0.21720 | 0.25172 | -0.03452 | -13.71% | +0.00000 | +0.00% | 0.03748 | 0.04977 | 0.12996 |
| earnings22 | test | teacher_kl | 1em7 | freq3_width24_time0 | rl_step_30000 | 0.21855 | 0.23007 | -0.01152 | -5.01% | +0.00135 | +0.62% | 0.04293 | 0.04207 | 0.13355 |
| earnings22 | test | teacher_kl | 1em7 | freq6_width34_time0 | old_seed | 0.23265 | 0.25172 | -0.01908 | -7.58% | +0.00000 | +0.00% | 0.04232 | 0.04849 | 0.14184 |
| earnings22 | test | teacher_kl | 1em7 | freq6_width34_time0 | rl_step_30000 | 0.23070 | 0.23007 | +0.00063 | +0.28% | -0.00194 | -0.83% | 0.04638 | 0.04160 | 0.14272 |
| earnings22 | test | teacher_kl | 3em7 | freq3_width24_time0 | old_seed | 0.22223 | 0.25172 | -0.02949 | -11.72% | +0.00000 | +0.00% | 0.03819 | 0.05132 | 0.13271 |
| earnings22 | test | teacher_kl | 3em7 | freq3_width24_time0 | rl_step_30000 | 0.21982 | 0.23007 | -0.01025 | -4.46% | -0.00241 | -1.08% | 0.04271 | 0.04391 | 0.13320 |
| earnings22 | test | teacher_kl | 3em7 | freq6_width34_time0 | old_seed | 0.23197 | 0.25172 | -0.01975 | -7.85% | +0.00000 | +0.00% | 0.04044 | 0.05014 | 0.14139 |
| earnings22 | test | teacher_kl | 3em7 | freq6_width34_time0 | rl_step_30000 | 0.23164 | 0.23007 | +0.00157 | +0.68% | -0.00033 | -0.14% | 0.04677 | 0.04242 | 0.14245 |
| rev16 | test | teacher_ce | 1em7 | freq3_width24_time0 | old_seed | 0.23790 | 0.17663 | +0.06127 | +34.69% | +0.00000 | +0.00% | 0.02323 | 0.14223 | 0.07243 |
| rev16 | test | teacher_ce | 1em7 | freq3_width24_time0 | rl_step_30000 | 0.23599 | 0.17206 | +0.06393 | +37.15% | -0.00191 | -0.80% | 0.02280 | 0.13812 | 0.07507 |
| rev16 | test | teacher_ce | 1em8 | freq1_width12_time0 | old_seed | 0.18423 | 0.17663 | +0.00760 | +4.30% | +0.00000 | +0.00% | 0.03376 | 0.06961 | 0.08086 |
| rev16 | test | teacher_ce | 1em8 | freq1_width12_time0 | rl_step_30000 | 0.48730 | 0.17206 | +0.31523 | +183.21% | +0.30307 | +164.51% | 0.18942 | 0.05230 | 0.24558 |
| rev16 | test | teacher_ce | 1em8 | freq3_width24_time0 | old_seed | 0.23744 | 0.17663 | +0.06081 | +34.43% | +0.00000 | +0.00% | 0.02315 | 0.14202 | 0.07227 |
| rev16 | test | teacher_ce | 1em8 | freq3_width24_time0 | rl_step_30000 | 0.17357 | 0.17206 | +0.00151 | +0.88% | -0.06387 | -26.90% | 0.03404 | 0.05714 | 0.08239 |
| rev16 | test | teacher_ce | 1em8 | no_aug | old_seed | 0.25740 | 0.17663 | +0.08077 | +45.73% | +0.00000 | +0.00% | 0.01862 | 0.16927 | 0.06951 |
| rev16 | test | teacher_ce | 1em8 | no_aug | rl_step_30000 | 0.40760 | 0.17206 | +0.23554 | +136.89% | +0.15020 | +58.35% | 0.01246 | 0.34156 | 0.05358 |
| rev16 | test | teacher_ce | 3em8 | freq1_width12_time0 | old_seed | 0.24183 | 0.17663 | +0.06520 | +36.92% | +0.00000 | +0.00% | 0.02353 | 0.14489 | 0.07341 |
| rev16 | test | teacher_ce | 3em8 | freq1_width12_time0 | rl_step_30000 | 0.19263 | 0.17206 | +0.02057 | +11.95% | -0.04920 | -20.34% | 0.03622 | 0.06471 | 0.09171 |
| rev16 | test | teacher_ce | 3em8 | freq3_width24_time0 | old_seed | 0.17915 | 0.17663 | +0.00252 | +1.43% | +0.00000 | +0.00% | 0.03636 | 0.06193 | 0.08086 |
| rev16 | test | teacher_ce | 3em8 | freq3_width24_time0 | rl_step_30000 | 0.31086 | 0.17206 | +0.13879 | +80.66% | +0.13170 | +73.51% | 0.01956 | 0.22675 | 0.06455 |
| rev16 | test | teacher_ce | 3em8 | no_aug | old_seed | 0.24803 | 0.17663 | +0.07140 | +40.42% | +0.00000 | +0.00% | 0.01977 | 0.15787 | 0.07038 |
| rev16 | test | teacher_ce | 3em8 | no_aug | rl_step_30000 | 0.39313 | 0.17206 | +0.22107 | +128.48% | +0.14510 | +58.50% | 0.01093 | 0.33274 | 0.04947 |
| rev16 | test | teacher_ce | 3em9 | freq3_width24_time0 | old_seed | 0.37051 | 0.17663 | +0.19388 | +109.77% | +0.00000 | +0.00% | 0.07735 | 0.14591 | 0.14725 |
| rev16 | test | teacher_ce | 3em9 | freq3_width24_time0 | rl_step_30000 | 0.39903 | 0.17206 | +0.22697 | +131.91% | +0.02852 | +7.70% | 0.07600 | 0.18076 | 0.14227 |
| tedlium | test | teacher_ce | 1em7 | freq3_width24_time0 | old_seed | 0.08811 | 0.08896 | -0.00085 | -0.96% | +0.00000 | +0.00% | 0.01240 | 0.03406 | 0.04164 |
| tedlium | test | teacher_ce | 1em7 | freq3_width24_time0 | rl_step_30000 | 0.07992 | 0.08386 | -0.00393 | -4.69% | -0.00819 | -9.29% | 0.01067 | 0.02764 | 0.04161 |
| tedlium | test | teacher_ce | 1em7 | freq6_width34_time0 | old_seed | 0.21705 | 0.08896 | +0.12809 | +143.98% | +0.00000 | +0.00% | 0.01007 | 0.16867 | 0.03831 |
| tedlium | test | teacher_ce | 1em7 | freq6_width34_time0 | rl_step_30000 | 0.08286 | 0.08386 | -0.00099 | -1.18% | -0.13418 | -61.82% | 0.01205 | 0.02867 | 0.04214 |
| tedlium | test | teacher_ce | 3em7 | freq3_width24_time0 | old_seed | 0.08747 | 0.08896 | -0.00149 | -1.67% | +0.00000 | +0.00% | 0.01262 | 0.03363 | 0.04122 |
| tedlium | test | teacher_ce | 3em7 | freq3_width24_time0 | rl_step_30000 | 0.08173 | 0.08386 | -0.00213 | -2.54% | -0.00574 | -6.56% | 0.01042 | 0.02878 | 0.04253 |
| tedlium | test | teacher_ce | 3em7 | freq6_width34_time0 | old_seed | 0.08871 | 0.08896 | -0.00025 | -0.28% | +0.00000 | +0.00% | 0.01109 | 0.03573 | 0.04189 |
| tedlium | test | teacher_ce | 3em7 | freq6_width34_time0 | rl_step_30000 | 0.08414 | 0.08386 | +0.00028 | +0.34% | -0.00457 | -5.15% | 0.01106 | 0.03094 | 0.04214 |
| tedlium | test | teacher_kl | 1em7 | freq3_width24_time0 | old_seed | 0.09374 | 0.08896 | +0.00478 | +5.38% | +0.00000 | +0.00% | 0.01882 | 0.03222 | 0.04271 |
| tedlium | test | teacher_kl | 1em7 | freq3_width24_time0 | rl_step_30000 | 0.08389 | 0.08386 | +0.00004 | +0.04% | -0.00985 | -10.51% | 0.01155 | 0.02956 | 0.04278 |
| tedlium | test | teacher_kl | 1em7 | freq6_width34_time0 | old_seed | 0.09236 | 0.08896 | +0.00340 | +3.82% | +0.00000 | +0.00% | 0.01705 | 0.03215 | 0.04317 |
| tedlium | test | teacher_kl | 1em7 | freq6_width34_time0 | rl_step_30000 | 0.08350 | 0.08386 | -0.00035 | -0.42% | -0.00886 | -9.59% | 0.01163 | 0.02949 | 0.04239 |
| tedlium | test | teacher_kl | 3em7 | freq3_width24_time0 | old_seed | 0.09431 | 0.08896 | +0.00535 | +6.02% | +0.00000 | +0.00% | 0.01875 | 0.03204 | 0.04352 |
| tedlium | test | teacher_kl | 3em7 | freq3_width24_time0 | rl_step_30000 | 0.08542 | 0.08386 | +0.00156 | +1.86% | -0.00890 | -9.43% | 0.01389 | 0.02874 | 0.04278 |
| tedlium | test | teacher_kl | 3em7 | freq6_width34_time0 | old_seed | 0.09477 | 0.08896 | +0.00581 | +6.53% | +0.00000 | +0.00% | 0.01992 | 0.03190 | 0.04296 |
| tedlium | test | teacher_kl | 3em7 | freq6_width34_time0 | rl_step_30000 | 0.08379 | 0.08386 | -0.00007 | -0.08% | -0.01099 | -11.59% | 0.01088 | 0.03020 | 0.04271 |
