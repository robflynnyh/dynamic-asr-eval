# ROB-63 RL self-training comparison

Snapshot generated from completed ROB-63 pickles.
RL deltas compare each RL `step_30000` self-training cell against the matching old-seed cell.
Normal/unadapted deltas compare each self-training cell against the same checkpoint's normal decoding WER.

## Checkpoint Key

- `old_seed`: normal encoder-decoder seed checkpoint at `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`.
- `rl_step_30000`: 30K RL-trained checkpoint from the ROB-61/PR #11 lineage at `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt`.

Every row in the full table also carries the exact `checkpoint_path` used for that row.

## Summary

- Completed 12 one-epoch result rows covering 2 datasets, 2 splits, 2 checkpoints, 1 training modes, 3 learning rates, 1 frequency-mask settings.
- The sweep used no teacher filtering. `Delta vs old seed` is only meaningful for the RL rows because old-seed rows are the matching-cell reference.
- chime6/dev: RL `step_30000` beats the matching old-seed cell in 0/3 cells; RL-vs-old absolute WER deltas span +0.00000 to +0.45427. Best RL cell is rl_step_30000 teacher_ce lr=1em8 freq3_width24_time0 at 1.00000 WER; best old-seed cell is old_seed teacher_ce lr=1em8 freq3_width24_time0 at 1.00000 WER. Relative change vs normal decoding spans +23.22% to +79.19% for RL and +19.85% to +19.85% for old seed.
- rev16/test: RL `step_30000` beats the matching old-seed cell in 1/3 cells; RL-vs-old absolute WER deltas span -0.06387 to +0.13170. Best RL cell is rl_step_30000 teacher_ce lr=1em8 freq3_width24_time0 at 0.17357 WER; best old-seed cell is old_seed teacher_ce lr=3em8 freq3_width24_time0 at 0.17915 WER. Relative change vs normal decoding spans +0.88% to +131.91% for RL and +1.43% to +109.77% for old seed.
- Sanity note: rev16 rl_step_30000 teacher_ce lr=3em9 freq3_width24_time0 is an outlier versus normal decoding (+131.91%, WER 0.39903); the error mix is dominated by deletion rate 0.18076.
- Sanity note: rev16 old_seed teacher_ce lr=3em9 freq3_width24_time0 is an outlier versus normal decoding (+109.77%, WER 0.37051); the error mix is dominated by deletion rate 0.14591.
- Sanity note: rev16 rl_step_30000 teacher_ce lr=3em8 freq3_width24_time0 is an outlier versus normal decoding (+80.66%, WER 0.31086); the error mix is dominated by deletion rate 0.22675.
- Sanity note: chime6 rl_step_30000 teacher_ce lr=3em9 freq3_width24_time0 is an outlier versus normal decoding (+79.19%, WER 1.45427); the error mix is dominated by deletion rate 0.62108.
- Sanity note: rev16 old_seed teacher_ce lr=1em8 freq3_width24_time0 is an outlier versus normal decoding (+34.43%, WER 0.23744); the error mix is dominated by deletion rate 0.14202.

## Unadapted vs adapted WER

This is the main readout. `Old normal WER` uses the `old_seed` checkpoint from the checkpoint key above, and `RL normal WER` uses the `rl_step_30000` checkpoint from the same key. Both are unadapted beam5/lp0.5 decoding baselines for the matching checkpoint. The adapted columns are the one-epoch self-training WERs for the listed setting.

| Dataset | Split | Mode | LR | Augmentation | Old normal WER | Old adapted WER | Old adapted vs normal | RL normal WER | RL adapted WER | RL adapted vs normal | RL adapted vs old adapted |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| chime6 | dev | teacher_ce | 1em8 | freq3_width24_time0 | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 1.00000 | +0.18843 | +0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq3_width24_time0 | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 1.00000 | +0.18843 | +0.00000 |
| chime6 | dev | teacher_ce | 3em9 | freq3_width24_time0 | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 1.45427 | +0.64269 | +0.45427 |
| rev16 | test | teacher_ce | 1em8 | freq3_width24_time0 | 0.17663 | 0.23744 | +0.06081 | 0.17206 | 0.17357 | +0.00151 | -0.06387 |
| rev16 | test | teacher_ce | 3em8 | freq3_width24_time0 | 0.17663 | 0.17915 | +0.00252 | 0.17206 | 0.31086 | +0.13879 | +0.13170 |
| rev16 | test | teacher_ce | 3em9 | freq3_width24_time0 | 0.17663 | 0.37051 | +0.19388 | 0.17206 | 0.39903 | +0.22697 | +0.02852 |

## Full table

| Dataset | Split | Mode | LR | Augmentation | Checkpoint | Checkpoint path | Adapted WER | Unadapted WER | Delta vs unadapted | Relative vs unadapted | Delta vs old seed | Relative vs old seed | Ins | Del | Sub |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chime6 | dev | teacher_ce | 1em8 | freq3_width24_time0 | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | freq3_width24_time0 | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 1.00000 | 0.81157 | +0.18843 | +23.22% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq3_width24_time0 | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq3_width24_time0 | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 1.00000 | 0.81157 | +0.18843 | +23.22% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em9 | freq3_width24_time0 | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em9 | freq3_width24_time0 | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 1.45427 | 0.81157 | +0.64269 | +79.19% | +0.45427 | +45.43% | 0.49376 | 0.62108 | 0.33942 |
| rev16 | test | teacher_ce | 1em8 | freq3_width24_time0 | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 0.23744 | 0.17663 | +0.06081 | +34.43% | +0.00000 | +0.00% | 0.02315 | 0.14202 | 0.07227 |
| rev16 | test | teacher_ce | 1em8 | freq3_width24_time0 | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 0.17357 | 0.17206 | +0.00151 | +0.88% | -0.06387 | -26.90% | 0.03404 | 0.05714 | 0.08239 |
| rev16 | test | teacher_ce | 3em8 | freq3_width24_time0 | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 0.17915 | 0.17663 | +0.00252 | +1.43% | +0.00000 | +0.00% | 0.03636 | 0.06193 | 0.08086 |
| rev16 | test | teacher_ce | 3em8 | freq3_width24_time0 | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 0.31086 | 0.17206 | +0.13879 | +80.66% | +0.13170 | +73.51% | 0.01956 | 0.22675 | 0.06455 |
| rev16 | test | teacher_ce | 3em9 | freq3_width24_time0 | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 0.37051 | 0.17663 | +0.19388 | +109.77% | +0.00000 | +0.00% | 0.07735 | 0.14591 | 0.14725 |
| rev16 | test | teacher_ce | 3em9 | freq3_width24_time0 | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 0.39903 | 0.17206 | +0.22697 | +131.91% | +0.02852 | +7.70% | 0.07600 | 0.18076 | 0.14227 |
