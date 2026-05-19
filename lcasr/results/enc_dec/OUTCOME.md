# Encoder-decoder ROB-63 checkpoint outcome

This top-level readout is intentionally checkpoint-focused. It shows the best
completed adapted row for the newest checkpoint and the matched old-seed row
for the same dataset, split, mode, learning rate, augmentation, decode, and
epoch. All rows are single-repeat snapshots.

## Checkpoint Folders

| Folder | Checkpoint | Meaning |
|---|---|---|
| `enc_dec_v2/` | `enc_dec_v2` | Historical encoder-decoder checkpoint at `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt`; this is the older family with higher Earnings22 unadapted WER. |
| `old_seed/` | `old_seed` | normal encoder-decoder seed checkpoint at `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` |
| `rl_step_30000/` | `rl_step_30000` | 30K RL-trained checkpoint from the ROB-61/PR #11 lineage at `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` |

Checkpoint-specific ROB-63 summaries live in `old_seed/OUTCOME.md` and
`rl_step_30000/OUTCOME.md`. Historical enc-dec-v2 outputs live in
`enc_dec_v2/`. The full paired ROB-63 table remains in
`rl_step_30000/rob63_rl_self_training_compare/COMBINED_OUTCOME.md`.

## Best Newest-Checkpoint Rows

`rl_step_30000` is the newest RL-trained checkpoint. `old_seed` is the
normal ROB-63 encoder-decoder seed checkpoint. Baselines are unadapted beam5/lp0.5
decodes for each checkpoint.

| Dataset | Split | Best rl_step_30000 setting | Old-seed normal WER | Old-seed matched adapted WER | rl_step_30000 normal WER | rl_step_30000 best adapted WER | rl_step_30000 adapted vs normal | rl_step_30000 adapted vs old-seed adapted |
|---|---|---|---:|---:|---:|---:|---:|---:|
| chime6 | dev | `teacher_ce lr=1em8 freq9_width44_time0` | 0.83439 | 1.00000 | 0.81157 | 0.99413 | +0.18255 | -0.00587 |
| chime6 | test | `teacher_ce lr=1em7 freq3_width24_time0` | 0.86477 | 1.00000 | 0.85160 | 1.00000 | +0.14840 | +0.00000 |
| earnings22 | test | `teacher_ce lr=1em7 freq3_width24_time0` | 0.25172 | 0.21806 | 0.23007 | 0.21365 | -0.01642 | -0.00441 |
| rev16 | test | `teacher_ce lr=3em8 freq9_width44_time0` | 0.17663 | 0.17446 | 0.17206 | 0.17176 | -0.00030 | -0.00270 |
| tedlium | test | `teacher_ce lr=1em7 freq3_width24_time0` | 0.08896 | 0.08811 | 0.08386 | 0.07992 | -0.00393 | -0.00819 |

## Interpretation

- TED-LIUM and Earnings22 are the cleanest wins for the RL checkpoint under
  the completed one-epoch self-training grid.
- Rev16 is sensitive to learning rate and masking: the best RL row is a small
  gain over both its own normal decode and the matched old-seed adapted row,
  but nearby higher-LR cells collapse.
- CHiME-6 remains deletion-dominated or near 1.0 WER after adaptation, so it
  should be treated as a failed adaptation setting rather than a checkpoint win.
- The `enc_dec_v2/` folder is a separate historical checkpoint family using
  `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt`; use `old_seed/` and `rl_step_30000/` for
  ROB-63 seed-vs-RL comparisons.
