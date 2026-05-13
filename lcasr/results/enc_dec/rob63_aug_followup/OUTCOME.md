# ROB-63 RL self-training comparison

Snapshot generated from completed ROB-63 pickles.
RL deltas compare each RL `step_30000` self-training cell against the matching old-seed cell.
Normal/unadapted deltas compare each self-training cell against the same checkpoint's normal decoding WER.

## Checkpoint Key

- `old_seed`: normal encoder-decoder seed checkpoint at `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`.
- `rl_step_30000`: 30K RL-trained checkpoint from the ROB-61/PR #11 lineage at `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt`.

Every row in the full table also carries the exact `checkpoint_path` used for that row.

## Summary

- Completed 16 one-epoch result rows covering 2 datasets, 2 splits, 2 checkpoints, 1 training modes, 2 learning rates, 2 frequency-mask settings.
- The sweep used no teacher filtering. `Delta vs old seed` is only meaningful for the RL rows because old-seed rows are the matching-cell reference.
- chime6/dev: RL `step_30000` beats the matching old-seed cell in 0/4 cells; RL-vs-old absolute WER deltas span +0.00000 to +0.00000. Best RL cell is rl_step_30000 teacher_ce lr=1em8 freq1_width12_time0 at 1.00000 WER; best old-seed cell is old_seed teacher_ce lr=1em8 freq1_width12_time0 at 1.00000 WER. Relative change vs normal decoding spans +23.22% to +23.22% for RL and +19.85% to +19.85% for old seed.
- rev16/test: RL `step_30000` beats the matching old-seed cell in 1/4 cells; RL-vs-old absolute WER deltas span -0.04920 to +0.30307. Best RL cell is rl_step_30000 teacher_ce lr=3em8 freq1_width12_time0 at 0.19263 WER; best old-seed cell is old_seed teacher_ce lr=1em8 freq1_width12_time0 at 0.18423 WER. Relative change vs normal decoding spans +11.95% to +183.21% for RL and +4.30% to +45.73% for old seed.
- Sanity note: rev16 rl_step_30000 teacher_ce lr=1em8 freq1_width12_time0 is an outlier versus normal decoding (+183.21%, WER 0.48730); the error mix is dominated by deletion rate 0.05230.
- Sanity note: rev16 rl_step_30000 teacher_ce lr=1em8 no_aug is an outlier versus normal decoding (+136.89%, WER 0.40760); the error mix is dominated by deletion rate 0.34156.
- Sanity note: rev16 rl_step_30000 teacher_ce lr=3em8 no_aug is an outlier versus normal decoding (+128.48%, WER 0.39313); the error mix is dominated by deletion rate 0.33274.
- Sanity note: rev16 old_seed teacher_ce lr=1em8 no_aug is an outlier versus normal decoding (+45.73%, WER 0.25740); the error mix is dominated by deletion rate 0.16927.
- Sanity note: rev16 old_seed teacher_ce lr=3em8 no_aug is an outlier versus normal decoding (+40.42%, WER 0.24803); the error mix is dominated by deletion rate 0.15787.
- Sanity note: rev16 old_seed teacher_ce lr=3em8 freq1_width12_time0 is an outlier versus normal decoding (+36.92%, WER 0.24183); the error mix is dominated by deletion rate 0.14489.

## Unadapted vs adapted WER

This is the main readout. `Old normal WER` uses the `old_seed` checkpoint from the checkpoint key above, and `RL normal WER` uses the `rl_step_30000` checkpoint from the same key. Both are unadapted beam5/lp0.5 decoding baselines for the matching checkpoint. The adapted columns are the one-epoch self-training WERs for the listed setting.

| Dataset | Split | Mode | LR | Augmentation | Old normal WER | Old adapted WER | Old adapted vs normal | RL normal WER | RL adapted WER | RL adapted vs normal | RL adapted vs old adapted |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| chime6 | dev | teacher_ce | 1em8 | freq1_width12_time0 | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 1.00000 | +0.18843 | +0.00000 |
| chime6 | dev | teacher_ce | 1em8 | no_aug | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 1.00000 | +0.18843 | +0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq1_width12_time0 | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 1.00000 | +0.18843 | +0.00000 |
| chime6 | dev | teacher_ce | 3em8 | no_aug | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 1.00000 | +0.18843 | +0.00000 |
| rev16 | test | teacher_ce | 1em8 | freq1_width12_time0 | 0.17663 | 0.18423 | +0.00760 | 0.17206 | 0.48730 | +0.31523 | +0.30307 |
| rev16 | test | teacher_ce | 1em8 | no_aug | 0.17663 | 0.25740 | +0.08077 | 0.17206 | 0.40760 | +0.23554 | +0.15020 |
| rev16 | test | teacher_ce | 3em8 | freq1_width12_time0 | 0.17663 | 0.24183 | +0.06520 | 0.17206 | 0.19263 | +0.02057 | -0.04920 |
| rev16 | test | teacher_ce | 3em8 | no_aug | 0.17663 | 0.24803 | +0.07140 | 0.17206 | 0.39313 | +0.22107 | +0.14510 |

## Full table

| Dataset | Split | Mode | LR | Augmentation | Checkpoint | Checkpoint path | Adapted WER | Unadapted WER | Delta vs unadapted | Relative vs unadapted | Delta vs old seed | Relative vs old seed | Ins | Del | Sub |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chime6 | dev | teacher_ce | 1em8 | freq1_width12_time0 | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | freq1_width12_time0 | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 1.00000 | 0.81157 | +0.18843 | +23.22% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | no_aug | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | no_aug | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 1.00000 | 0.81157 | +0.18843 | +23.22% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq1_width12_time0 | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | freq1_width12_time0 | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 1.00000 | 0.81157 | +0.18843 | +23.22% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | no_aug | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 3em8 | no_aug | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 1.00000 | 0.81157 | +0.18843 | +23.22% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| rev16 | test | teacher_ce | 1em8 | freq1_width12_time0 | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 0.18423 | 0.17663 | +0.00760 | +4.30% | +0.00000 | +0.00% | 0.03376 | 0.06961 | 0.08086 |
| rev16 | test | teacher_ce | 1em8 | freq1_width12_time0 | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 0.48730 | 0.17206 | +0.31523 | +183.21% | +0.30307 | +164.51% | 0.18942 | 0.05230 | 0.24558 |
| rev16 | test | teacher_ce | 1em8 | no_aug | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 0.25740 | 0.17663 | +0.08077 | +45.73% | +0.00000 | +0.00% | 0.01862 | 0.16927 | 0.06951 |
| rev16 | test | teacher_ce | 1em8 | no_aug | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 0.40760 | 0.17206 | +0.23554 | +136.89% | +0.15020 | +58.35% | 0.01246 | 0.34156 | 0.05358 |
| rev16 | test | teacher_ce | 3em8 | freq1_width12_time0 | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 0.24183 | 0.17663 | +0.06520 | +36.92% | +0.00000 | +0.00% | 0.02353 | 0.14489 | 0.07341 |
| rev16 | test | teacher_ce | 3em8 | freq1_width12_time0 | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 0.19263 | 0.17206 | +0.02057 | +11.95% | -0.04920 | -20.34% | 0.03622 | 0.06471 | 0.09171 |
| rev16 | test | teacher_ce | 3em8 | no_aug | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 0.24803 | 0.17663 | +0.07140 | +40.42% | +0.00000 | +0.00% | 0.01977 | 0.15787 | 0.07038 |
| rev16 | test | teacher_ce | 3em8 | no_aug | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 0.39313 | 0.17206 | +0.22107 | +128.48% | +0.14510 | +58.50% | 0.01093 | 0.33274 | 0.04947 |
