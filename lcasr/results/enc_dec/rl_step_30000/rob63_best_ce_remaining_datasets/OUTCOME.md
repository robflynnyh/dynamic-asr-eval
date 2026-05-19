# ROB-63 RL self-training comparison

Snapshot generated from completed ROB-63 pickles.
RL deltas compare each RL `step_30000` self-training cell against the matching old-seed cell.
Normal/unadapted deltas compare each self-training cell against the same checkpoint's normal decoding WER.

## Checkpoint Key

- `old_seed`: normal encoder-decoder seed checkpoint at `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`.
- `rl_step_30000`: 30K RL-trained checkpoint from the ROB-61/PR #11 lineage at `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt`.

Every row in the full table also carries the exact `checkpoint_path` used for that row.

## Summary

- Completed 4 one-epoch result rows covering 2 datasets, 1 splits, 2 checkpoints, 1 training modes, 1 learning rates, 1 frequency-mask settings.
- The sweep used no teacher filtering. `Delta vs old seed` is only meaningful for the RL rows because old-seed rows are the matching-cell reference.
- chime6/test: RL `step_30000` beats the matching old-seed cell in 0/1 cells; RL-vs-old absolute WER deltas span +0.00000 to +0.00000. Best RL cell is rl_step_30000 teacher_ce lr=1em7 freq3_width24_time0 at 1.00000 WER; best old-seed cell is old_seed teacher_ce lr=1em7 freq3_width24_time0 at 1.00000 WER. Relative change vs normal decoding spans +17.43% to +17.43% for RL and +15.64% to +15.64% for old seed.
- rev16/test: RL `step_30000` beats the matching old-seed cell in 1/1 cells; RL-vs-old absolute WER deltas span -0.00191 to -0.00191. Best RL cell is rl_step_30000 teacher_ce lr=1em7 freq3_width24_time0 at 0.23599 WER; best old-seed cell is old_seed teacher_ce lr=1em7 freq3_width24_time0 at 0.23790 WER. Relative change vs normal decoding spans +37.15% to +37.15% for RL and +34.69% to +34.69% for old seed.
- Sanity note: rev16 rl_step_30000 teacher_ce lr=1em7 freq3_width24_time0 is an outlier versus normal decoding (+37.15%, WER 0.23599); the error mix is dominated by deletion rate 0.13812.
- Sanity note: rev16 old_seed teacher_ce lr=1em7 freq3_width24_time0 is an outlier versus normal decoding (+34.69%, WER 0.23790); the error mix is dominated by deletion rate 0.14223.

## Unadapted vs adapted WER

This is the main readout. `Old normal WER` uses the `old_seed` checkpoint from the checkpoint key above, and `RL normal WER` uses the `rl_step_30000` checkpoint from the same key. Both are unadapted beam5/lp0.5 decoding baselines for the matching checkpoint. The adapted columns are the one-epoch self-training WERs for the listed setting.

| Dataset | Split | Mode | LR | Augmentation | Old normal WER | Old adapted WER | Old adapted vs normal | RL normal WER | RL adapted WER | RL adapted vs normal | RL adapted vs old adapted |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| chime6 | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.86477 | 1.00000 | +0.13523 | 0.85160 | 1.00000 | +0.14840 | +0.00000 |
| rev16 | test | teacher_ce | 1em7 | freq3_width24_time0 | 0.17663 | 0.23790 | +0.06127 | 0.17206 | 0.23599 | +0.06393 | -0.00191 |

## Full table

| Dataset | Split | Mode | LR | Augmentation | Checkpoint | Checkpoint path | Adapted WER | Unadapted WER | Delta vs unadapted | Relative vs unadapted | Delta vs old seed | Relative vs old seed | Ins | Del | Sub |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chime6 | test | teacher_ce | 1em7 | freq3_width24_time0 | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 1.00000 | 0.86477 | +0.13523 | +15.64% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | test | teacher_ce | 1em7 | freq3_width24_time0 | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 1.00000 | 0.85160 | +0.14840 | +17.43% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| rev16 | test | teacher_ce | 1em7 | freq3_width24_time0 | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 0.23790 | 0.17663 | +0.06127 | +34.69% | +0.00000 | +0.00% | 0.02323 | 0.14223 | 0.07243 |
| rev16 | test | teacher_ce | 1em7 | freq3_width24_time0 | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 0.23599 | 0.17206 | +0.06393 | +37.15% | -0.00191 | -0.80% | 0.02280 | 0.13812 | 0.07507 |
