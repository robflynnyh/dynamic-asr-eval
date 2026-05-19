# Checkpoint 1: old_seed

`old_seed` is the normal encoder-decoder seed checkpoint.

Checkpoint path: `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`

Rows are one-epoch self-training snapshots. `Normal WER` is the matching
unadapted beam5/lp0.5 decode for the same checkpoint when available.

## Best Rows

| Dataset | Split | Mode | LR | Augmentation | Normal WER | Adapted WER | Adapted vs normal |
|---|---|---|---:|---|---:|---:|---:|
| chime6 | dev | teacher_ce | 3em8 | freq9_width44_time0 | 0.83439 | 0.99411 | +0.15972 |
| chime6 | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.86477 | 1.00000 | +0.13523 |
| earnings22 | test | teacher_ce | 3em7 | freq3_width24_time0 | 0.25172 | 0.21686 | -0.03486 |
| rev16 | test | teacher_ce | 3em8 | freq9_width44_time0 | 0.17663 | 0.17446 | -0.00217 |
| tedlium | test | teacher_ce | 3em7 | freq3_width24_time0 | 0.08896 | 0.08747 | -0.00149 |

## Full Checkpoint Table

| Dataset | Split | Mode | LR | Augmentation | Adapted WER | Normal WER | Delta vs normal | Relative vs normal | Ins | Del | Sub |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| chime6 | dev | teacher_ce | 1em8 | freq1_width12_time0 | 1.00000 | 0.83439 | +0.16561 | +19.85% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | freq3_width24_time0 | 1.00000 | 0.83439 | +0.16561 | +19.85% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | freq9_width44_time0 | 1.00000 | 0.83439 | +0.16561 | +19.85% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | freq9_width44_time0_basic_repeat_filter | 1.00000 | 0.83439 | +0.16561 | +19.85% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | no_aug | 1.00000 | 0.83439 | +0.16561 | +19.85% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq1_width12_time0 | 1.00000 | 0.83439 | +0.16561 | +19.85% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq3_width24_time0 | 1.00000 | 0.83439 | +0.16561 | +19.85% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq9_width44_time0 | 0.99411 | 0.83439 | +0.15972 | +19.14% | 0.00000 | 0.99339 | 0.00072 |
| chime6 | dev | teacher_ce | 3em8 | freq9_width44_time0_basic_repeat_filter | 0.99514 | 0.83439 | +0.16075 | +19.27% | 0.00002 | 0.99444 | 0.00069 |
| chime6 | dev | teacher_ce | 3em8 | no_aug | 1.00000 | 0.83439 | +0.16561 | +19.85% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em9 | freq3_width24_time0 | 1.00000 | 0.83439 | +0.16561 | +19.85% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | test | teacher_ce | 1em7 | freq3_width24_time0 | 1.00000 | 0.86477 | +0.13523 | +15.64% | 0.00000 | 1.00000 | 0.00000 |
| earnings22 | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.21806 | 0.25172 | -0.03366 | -13.37% | 0.03789 | 0.05292 | 0.12726 |
| earnings22 | test | teacher_ce | 1em7 | freq6_width34_time0 | 0.24604 | 0.25172 | -0.00568 | -2.26% | 0.05388 | 0.04881 | 0.14335 |
| earnings22 | test | teacher_ce | 3em7 | freq3_width24_time0 | 0.21686 | 0.25172 | -0.03486 | -13.85% | 0.04248 | 0.04902 | 0.12536 |
| earnings22 | test | teacher_ce | 3em7 | freq6_width34_time0 | 0.23630 | 0.25172 | -0.01542 | -6.13% | 0.04650 | 0.04934 | 0.14045 |
| earnings22 | test | teacher_ce | 3em8 | freq9_width44_time0 | 0.24929 | 0.25172 | -0.00243 | -0.97% | 0.04969 | 0.04761 | 0.15199 |
| earnings22 | test | teacher_kl | 1em7 | freq3_width24_time0 | 0.21720 | 0.25172 | -0.03452 | -13.71% | 0.03748 | 0.04977 | 0.12996 |
| earnings22 | test | teacher_kl | 1em7 | freq6_width34_time0 | 0.23265 | 0.25172 | -0.01908 | -7.58% | 0.04232 | 0.04849 | 0.14184 |
| earnings22 | test | teacher_kl | 3em7 | freq3_width24_time0 | 0.22223 | 0.25172 | -0.02949 | -11.72% | 0.03819 | 0.05132 | 0.13271 |
| earnings22 | test | teacher_kl | 3em7 | freq6_width34_time0 | 0.23197 | 0.25172 | -0.01975 | -7.85% | 0.04044 | 0.05014 | 0.14139 |
| rev16 | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.23790 | 0.17663 | +0.06127 | +34.69% | 0.02323 | 0.14223 | 0.07243 |
| rev16 | test | teacher_ce | 1em7 | freq9_width44_time0 | 0.17507 | 0.17663 | -0.00156 | -0.88% | 0.03012 | 0.06309 | 0.08187 |
| rev16 | test | teacher_ce | 1em8 | freq1_width12_time0 | 0.18423 | 0.17663 | +0.00760 | +4.30% | 0.03376 | 0.06961 | 0.08086 |
| rev16 | test | teacher_ce | 1em8 | freq3_width24_time0 | 0.23744 | 0.17663 | +0.06081 | +34.43% | 0.02315 | 0.14202 | 0.07227 |
| rev16 | test | teacher_ce | 1em8 | freq9_width44_time0 | 0.24006 | 0.17663 | +0.06343 | +35.91% | 0.02233 | 0.14358 | 0.07415 |
| rev16 | test | teacher_ce | 1em8 | freq9_width44_time0_basic_repeat_filter | 0.23693 | 0.17663 | +0.06030 | +34.14% | 0.01925 | 0.14502 | 0.07265 |
| rev16 | test | teacher_ce | 1em8 | no_aug | 0.25740 | 0.17663 | +0.08077 | +45.73% | 0.01862 | 0.16927 | 0.06951 |
| rev16 | test | teacher_ce | 3em8 | freq1_width12_time0 | 0.24183 | 0.17663 | +0.06520 | +36.92% | 0.02353 | 0.14489 | 0.07341 |
| rev16 | test | teacher_ce | 3em8 | freq3_width24_time0 | 0.17915 | 0.17663 | +0.00252 | +1.43% | 0.03636 | 0.06193 | 0.08086 |
| rev16 | test | teacher_ce | 3em8 | freq9_width44_time0 | 0.17446 | 0.17663 | -0.00217 | -1.23% | 0.02987 | 0.06226 | 0.08234 |
| rev16 | test | teacher_ce | 3em8 | freq9_width44_time0_basic_repeat_filter | 0.17524 | 0.17663 | -0.00139 | -0.79% | 0.02976 | 0.06302 | 0.08245 |
| rev16 | test | teacher_ce | 3em8 | no_aug | 0.24803 | 0.17663 | +0.07140 | +40.42% | 0.01977 | 0.15787 | 0.07038 |
| rev16 | test | teacher_ce | 3em9 | freq3_width24_time0 | 0.37051 | 0.17663 | +0.19388 | +109.77% | 0.07735 | 0.14591 | 0.14725 |
| tedlium | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.08811 | 0.08896 | -0.00085 | -0.96% | 0.01240 | 0.03406 | 0.04164 |
| tedlium | test | teacher_ce | 1em7 | freq6_width34_time0 | 0.21705 | 0.08896 | +0.12809 | +143.98% | 0.01007 | 0.16867 | 0.03831 |
| tedlium | test | teacher_ce | 3em7 | freq3_width24_time0 | 0.08747 | 0.08896 | -0.00149 | -1.67% | 0.01262 | 0.03363 | 0.04122 |
| tedlium | test | teacher_ce | 3em7 | freq6_width34_time0 | 0.08871 | 0.08896 | -0.00025 | -0.28% | 0.01109 | 0.03573 | 0.04189 |
| tedlium | test | teacher_kl | 1em7 | freq3_width24_time0 | 0.09374 | 0.08896 | +0.00478 | +5.38% | 0.01882 | 0.03222 | 0.04271 |
| tedlium | test | teacher_kl | 1em7 | freq6_width34_time0 | 0.09236 | 0.08896 | +0.00340 | +3.82% | 0.01705 | 0.03215 | 0.04317 |
| tedlium | test | teacher_kl | 3em7 | freq3_width24_time0 | 0.09431 | 0.08896 | +0.00535 | +6.02% | 0.01875 | 0.03204 | 0.04352 |
| tedlium | test | teacher_kl | 3em7 | freq6_width34_time0 | 0.09477 | 0.08896 | +0.00581 | +6.53% | 0.01992 | 0.03190 | 0.04296 |
