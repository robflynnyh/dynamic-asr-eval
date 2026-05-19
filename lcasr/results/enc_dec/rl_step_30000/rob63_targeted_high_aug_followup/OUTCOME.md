# ROB-63 RL self-training comparison

Snapshot generated from completed ROB-63 pickles.
RL deltas compare each RL `step_30000` self-training cell against the matching old-seed cell.
Normal/unadapted deltas compare each self-training cell against the same checkpoint's normal decoding WER.

## Checkpoint Key

- `old_seed`: normal encoder-decoder seed checkpoint at `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`.
- `rl_step_30000`: 30K RL-trained checkpoint from the ROB-61/PR #11 lineage at `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt`.

Every row in the full table also carries the exact `checkpoint_path` used for that row.

## Summary

- Completed 4 one-epoch result rows covering 2 datasets, 1 splits, 2 checkpoints, 1 training modes, 2 learning rates, 1 frequency-mask settings.
- The sweep used no teacher filtering. `Delta vs old seed` is only meaningful for the RL rows because old-seed rows are the matching-cell reference.
- earnings22/test: RL `step_30000` beats the matching old-seed cell in 1/1 cells; RL-vs-old absolute WER deltas span -0.00188 to -0.00188. Best RL cell is rl_step_30000 teacher_ce lr=3em8 freq9_width44_time0 at 0.24741 WER; best old-seed cell is old_seed teacher_ce lr=3em8 freq9_width44_time0 at 0.24929 WER. Relative change vs normal decoding spans +7.54% to +7.54% for RL and -0.97% to -0.97% for old seed.
- rev16/test: RL `step_30000` beats the matching old-seed cell in 0/1 cells; RL-vs-old absolute WER deltas span +0.06121 to +0.06121. Best RL cell is rl_step_30000 teacher_ce lr=1em7 freq9_width44_time0 at 0.23628 WER; best old-seed cell is old_seed teacher_ce lr=1em7 freq9_width44_time0 at 0.17507 WER. Relative change vs normal decoding spans +37.32% to +37.32% for RL and -0.88% to -0.88% for old seed.
- Sanity note: rev16 rl_step_30000 teacher_ce lr=1em7 freq9_width44_time0 is an outlier versus normal decoding (+37.32%, WER 0.23628); the error mix is dominated by deletion rate 0.14114.

## Unadapted vs adapted WER

This is the main readout. `Old normal WER` uses the `old_seed` checkpoint from the checkpoint key above, and `RL normal WER` uses the `rl_step_30000` checkpoint from the same key. Both are unadapted beam5/lp0.5 decoding baselines for the matching checkpoint. The adapted columns are the one-epoch self-training WERs for the listed setting.

| Dataset | Split | Mode | LR | Augmentation | Old normal WER | Old adapted WER | Old adapted vs normal | RL normal WER | RL adapted WER | RL adapted vs normal | RL adapted vs old adapted |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| earnings22 | test | teacher_ce | 3em8 | freq9_width44_time0 | 0.25172 | 0.24929 | -0.00243 | 0.23007 | 0.24741 | +0.01734 | -0.00188 |
| rev16 | test | teacher_ce | 1em7 | freq9_width44_time0 | 0.17663 | 0.17507 | -0.00156 | 0.17206 | 0.23628 | +0.06421 | +0.06121 |

## Full table

| Dataset | Split | Mode | LR | Augmentation | Checkpoint | Checkpoint path | Adapted WER | Unadapted WER | Delta vs unadapted | Relative vs unadapted | Delta vs old seed | Relative vs old seed | Ins | Del | Sub |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| earnings22 | test | teacher_ce | 3em8 | freq9_width44_time0 | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 0.24929 | 0.25172 | -0.00243 | -0.97% | +0.00000 | +0.00% | 0.04969 | 0.04761 | 0.15199 |
| earnings22 | test | teacher_ce | 3em8 | freq9_width44_time0 | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 0.24741 | 0.23007 | +0.01734 | +7.54% | -0.00188 | -0.75% | 0.04957 | 0.04411 | 0.15373 |
| rev16 | test | teacher_ce | 1em7 | freq9_width44_time0 | old_seed | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` | 0.17507 | 0.17663 | -0.00156 | -0.88% | +0.00000 | +0.00% | 0.03012 | 0.06309 | 0.08187 |
| rev16 | test | teacher_ce | 1em7 | freq9_width44_time0 | rl_step_30000 | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` | 0.23628 | 0.17206 | +0.06421 | +37.32% | +0.06121 | +34.96% | 0.02091 | 0.14114 | 0.07423 |
