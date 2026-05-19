# Encoder-decoder results

This folder groups the encoder-decoder decoding and test-time adaptation result
sets.

## Layout

Only checkpoint folders should live directly under this directory:

| Folder | Purpose |
|---|---|
| `checkpoint1/` | Normal/non-RL encoder-decoder checkpoint results. The top-level files are the generated ROB-63 view for `old_seed`; nested folders hold the historical encoder-decoder seed ablations and the ROB-63 Earnings sanity check. |
| `checkpoint2/` | RL-checkpoint comparison results. The top-level files are the generated ROB-63 view for `rl_step_30000`; nested folders hold the ROB-61/ROB-63 seed-vs-RL benchmark and follow-up result sets. |

ROB-63 checkpoint mapping:

| Folder | Checkpoint key | Checkpoint path |
|---|---|---|
| `checkpoint1` | `old_seed` | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` |
| `checkpoint2` | `rl_step_30000` | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` |

The checkpoint folder top-level files are generated views. Nested folders keep
the raw result PKLs, logs, aggregators, and per-run outcomes.

Important nested folders:

| Folder | Location | Purpose |
|---|---|---|
| `enc_dec_beam_tedlium_dev` | `checkpoint1/` | Historical `enc_dec_v2` TEDLIUM dev beam-search decode sweep used to choose `beam=5`, `length_penalty=0.5`. |
| `enc_dec_dynamic_eval` | `checkpoint1/` | Historical `enc_dec_v2` teacher-CE dynamic-eval sweep with beam5/lp0.5 teacher and final decoding. |
| `enc_dec_teacher_kl*` | `checkpoint1/` | Historical `enc_dec_v2` teacher-KL sweeps, including entropy-filter and relaxed-filter variants. |
| `enc_dec_teacher_epoch_relabel` | `checkpoint1/` | Historical teacher relabel ablation. |
| `rob63_earnings_unadapted_sanity` | `checkpoint1/` | Earnings22 unadapted sanity check comparing `old_seed` with the older `enc_dec_v2` checkpoint family. |
| `rob61_checkpoint_benchmark` | `checkpoint2/` | Normal/unadapted baselines for `old_seed`, RL checkpoints, and the ROB-63 comparison deltas. |
| `rob63_rl_self_training_compare` | `checkpoint2/` | Main ROB-63 paired seed-vs-RL one-epoch self-training comparison. Use `checkpoint2/rob63_rl_self_training_compare/COMBINED_OUTCOME.md` for the full paired readout. |
| `rob63_*_followup` | `checkpoint2/` | ROB-63 auxiliary comparison follow-ups rolled into the combined ROB-63 summary. |

Most child folders contain:

```text
README.md
OUTCOME.md
aggregate.py
logs/
```

## Decode Comparability

Some folders evaluate greedy/default decoding, while others evaluate
beam-search decoding. Baselines are only comparable when the decode condition
matches.

The current adaptation folders mostly use:

```text
beam_width=5
length_penalty=0.5
```

The beam sweep folder intentionally compares many decode settings. Its WERs
measure decode-setting effects, not adaptation effects.

## Aggregation

Run any child aggregate from `lcasr/`:

```bash
python results/enc_dec/checkpoint1/enc_dec_beam_tedlium_dev/aggregate.py
python results/enc_dec/checkpoint1/enc_dec_dynamic_eval/aggregate.py
python results/enc_dec/checkpoint1/enc_dec_teacher_kl/aggregate.py
python results/enc_dec/checkpoint1/enc_dec_teacher_kl_entropy_filter/aggregate.py
python results/enc_dec/checkpoint1/enc_dec_teacher_kl_relaxed_filters/aggregate.py
python results/enc_dec/checkpoint1/enc_dec_teacher_epoch_relabel/aggregate.py
python results/enc_dec/checkpoint2/rob61_checkpoint_benchmark/aggregate.py --csv results/enc_dec/checkpoint2/rob61_checkpoint_benchmark/summary.csv --outcome results/enc_dec/checkpoint2/rob61_checkpoint_benchmark/OUTCOME.md
python results/enc_dec/checkpoint2/rob63_rl_self_training_compare/aggregate.py
python results/enc_dec/checkpoint2/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/checkpoint2/rob63_best_ce_remaining_datasets/pkl --csv results/enc_dec/checkpoint2/rob63_best_ce_remaining_datasets/summary.csv --outcome results/enc_dec/checkpoint2/rob63_best_ce_remaining_datasets/OUTCOME.md
python results/enc_dec/checkpoint2/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/checkpoint2/rob63_lower_lr_dev_followup/pkl --csv results/enc_dec/checkpoint2/rob63_lower_lr_dev_followup/summary.csv --outcome results/enc_dec/checkpoint2/rob63_lower_lr_dev_followup/OUTCOME.md
python results/enc_dec/checkpoint2/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/checkpoint2/rob63_aug_followup/pkl --csv results/enc_dec/checkpoint2/rob63_aug_followup/summary.csv --outcome results/enc_dec/checkpoint2/rob63_aug_followup/OUTCOME.md
python results/enc_dec/checkpoint2/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/checkpoint2/rob63_strong_aug_filter_followup/pkl --csv results/enc_dec/checkpoint2/rob63_strong_aug_filter_followup/summary.csv --outcome results/enc_dec/checkpoint2/rob63_strong_aug_filter_followup/OUTCOME.md
python results/enc_dec/checkpoint2/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/checkpoint2/rob63_targeted_high_aug_followup/pkl --csv results/enc_dec/checkpoint2/rob63_targeted_high_aug_followup/summary.csv --outcome results/enc_dec/checkpoint2/rob63_targeted_high_aug_followup/OUTCOME.md
python results/enc_dec/checkpoint2/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/checkpoint2/rob63_rl_self_training_compare/pkl --extra-directory results/enc_dec/checkpoint2/rob63_best_ce_remaining_datasets/pkl --extra-directory results/enc_dec/checkpoint2/rob63_lower_lr_dev_followup/pkl --extra-directory results/enc_dec/checkpoint2/rob63_aug_followup/pkl --extra-directory results/enc_dec/checkpoint2/rob63_strong_aug_filter_followup/pkl --extra-directory results/enc_dec/checkpoint2/rob63_targeted_high_aug_followup/pkl --csv results/enc_dec/checkpoint2/rob63_rl_self_training_compare/combined_summary.csv --outcome results/enc_dec/checkpoint2/rob63_rl_self_training_compare/COMBINED_OUTCOME.md --checkpoint-view-root results/enc_dec --top-level-outcome results/enc_dec/OUTCOME.md
```

To refresh CSV summaries:

```bash
python results/enc_dec/checkpoint1/enc_dec_dynamic_eval/aggregate.py --csv results/enc_dec/checkpoint1/enc_dec_dynamic_eval/summary.csv
python results/enc_dec/checkpoint1/enc_dec_teacher_kl/aggregate.py --csv results/enc_dec/checkpoint1/enc_dec_teacher_kl/summary.csv
python results/enc_dec/checkpoint1/enc_dec_teacher_kl_entropy_filter/aggregate.py --csv results/enc_dec/checkpoint1/enc_dec_teacher_kl_entropy_filter/summary.csv
python results/enc_dec/checkpoint1/enc_dec_teacher_kl_relaxed_filters/aggregate.py --csv results/enc_dec/checkpoint1/enc_dec_teacher_kl_relaxed_filters/summary.csv
python results/enc_dec/checkpoint1/enc_dec_teacher_epoch_relabel/aggregate.py --csv results/enc_dec/checkpoint1/enc_dec_teacher_epoch_relabel/summary.csv
python results/enc_dec/checkpoint2/rob61_checkpoint_benchmark/aggregate.py --csv results/enc_dec/checkpoint2/rob61_checkpoint_benchmark/summary.csv --outcome results/enc_dec/checkpoint2/rob61_checkpoint_benchmark/OUTCOME.md
python results/enc_dec/checkpoint2/rob63_rl_self_training_compare/aggregate.py --csv results/enc_dec/checkpoint2/rob63_rl_self_training_compare/summary.csv
python results/enc_dec/checkpoint2/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/checkpoint2/rob63_best_ce_remaining_datasets/pkl --csv results/enc_dec/checkpoint2/rob63_best_ce_remaining_datasets/summary.csv --outcome results/enc_dec/checkpoint2/rob63_best_ce_remaining_datasets/OUTCOME.md
python results/enc_dec/checkpoint2/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/checkpoint2/rob63_lower_lr_dev_followup/pkl --csv results/enc_dec/checkpoint2/rob63_lower_lr_dev_followup/summary.csv --outcome results/enc_dec/checkpoint2/rob63_lower_lr_dev_followup/OUTCOME.md
python results/enc_dec/checkpoint2/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/checkpoint2/rob63_aug_followup/pkl --csv results/enc_dec/checkpoint2/rob63_aug_followup/summary.csv --outcome results/enc_dec/checkpoint2/rob63_aug_followup/OUTCOME.md
python results/enc_dec/checkpoint2/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/checkpoint2/rob63_strong_aug_filter_followup/pkl --csv results/enc_dec/checkpoint2/rob63_strong_aug_filter_followup/summary.csv --outcome results/enc_dec/checkpoint2/rob63_strong_aug_filter_followup/OUTCOME.md
python results/enc_dec/checkpoint2/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/checkpoint2/rob63_targeted_high_aug_followup/pkl --csv results/enc_dec/checkpoint2/rob63_targeted_high_aug_followup/summary.csv --outcome results/enc_dec/checkpoint2/rob63_targeted_high_aug_followup/OUTCOME.md
python results/enc_dec/checkpoint2/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/checkpoint2/rob63_rl_self_training_compare/pkl --extra-directory results/enc_dec/checkpoint2/rob63_best_ce_remaining_datasets/pkl --extra-directory results/enc_dec/checkpoint2/rob63_lower_lr_dev_followup/pkl --extra-directory results/enc_dec/checkpoint2/rob63_aug_followup/pkl --extra-directory results/enc_dec/checkpoint2/rob63_strong_aug_filter_followup/pkl --extra-directory results/enc_dec/checkpoint2/rob63_targeted_high_aug_followup/pkl --csv results/enc_dec/checkpoint2/rob63_rl_self_training_compare/combined_summary.csv --outcome results/enc_dec/checkpoint2/rob63_rl_self_training_compare/COMBINED_OUTCOME.md --checkpoint-view-root results/enc_dec --top-level-outcome results/enc_dec/OUTCOME.md
```

## Launch Notes

Launch commands inside the child READMEs now point at nested
`results/enc_dec/checkpoint*/...` output folders. For new runs, keep
`RESULTS_DIR` under the relevant checkpoint folder when the launcher supports
it.
