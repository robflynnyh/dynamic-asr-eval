# ROB-63 rl_step_30000

This folder is the checkpoint-specific view of the ROB-63 encoder-decoder
self-training results. The top-level files are generated from the combined
ROB-63 result pickles and normal-decoding baselines. Nested folders under
`enc_dec_v2/`, `old_seed/`, and `rl_step_30000/` hold raw logs, PKLs,
and per-run outcomes grouped by checkpoint family.

## Checkpoint

- Folder: `results/enc_dec/rl_step_30000/`
- Key: `rl_step_30000`
- Description: 30K RL-trained checkpoint from the ROB-61/PR #11 lineage
- Path: `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt`

## Files

| File | Meaning |
|---|---|
| `summary.csv` | All ROB-63 adapted rows for this checkpoint, with matching normal/unadapted WER where available. |
| `OUTCOME.md` | Human-readable best-row summary plus the full checkpoint-specific table. |
| Nested result folders | Raw/result folders grouped by checkpoint family; only `enc_dec_v2/`, `old_seed/`, and `rl_step_30000/` should be direct children of `results/enc_dec/`. |

Regenerate from `lcasr/` with:

```bash
python results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py \
  --directory results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/pkl \
  --extra-directory results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/pkl \
  --extra-directory results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/pkl \
  --extra-directory results/enc_dec/rl_step_30000/rob63_aug_followup/pkl \
  --extra-directory results/enc_dec/rl_step_30000/rob63_strong_aug_filter_followup/pkl \
  --extra-directory results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/pkl \
  --csv results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/combined_summary.csv \
  --outcome results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/COMBINED_OUTCOME.md \
  --checkpoint-view-root results/enc_dec \
  --top-level-outcome results/enc_dec/OUTCOME.md
```
