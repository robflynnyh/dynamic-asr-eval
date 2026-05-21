# Encoder-decoder results

This folder groups the encoder-decoder decoding and test-time adaptation result
sets.

## Layout

Only checkpoint folders should live directly under this directory:

| Folder | Purpose |
|---|---|
| `enc_dec_v2/` | Historical encoder-decoder checkpoint family at `enc_dec_v2/step_105360.pt`; this is the older checkpoint with the higher Earnings22 unadapted WER. |
| `old_seed/` | ROB-63 normal encoder-decoder seed checkpoint view for `enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`. |
| `rl_step_30000/` | ROB-63 / ROB-61 RL-checkpoint comparison results for `rl_step_30000` and related paired seed-vs-RL artifacts. |

ROB-63 checkpoint mapping:

| Folder | Checkpoint key | Checkpoint path |
|---|---|---|
| `enc_dec_v2` | `enc_dec_v2` / `enc_dec_outcome_seed` | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt` |
| `old_seed` | `old_seed` | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` |
| `rl_step_30000` | `rl_step_30000` | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` |

The `old_seed/` and `rl_step_30000/` top-level files are generated ROB-63
checkpoint views. Nested folders keep raw result PKLs, logs, aggregators, and
per-run outcomes.

## Model Architecture

All three checkpoint families above use the same checkpoint architecture. The
details below were read from the checkpoint configs and verified by
instantiating the model from the `enc_dec_v2` checkpoint.

| Field | Value |
|---|---:|
| Model class | `EncDecSconformerV2` |
| Trainable parameters | 89,287,114 |
| Input acoustic features | 80 |
| Encoder layers | 3 |
| Decoder layers | 3 |
| Model width | 768 |
| Attention heads | 6 |
| Head dimension | 128 |
| Feed-forward expansion | 4 |
| Conformer convolution kernel | 9 |
| Subsampling | depthwise striding, factor 8 |
| CTC auxiliary loss weight | 0.05 |
| Acoustic norm | disabled |
| Self-conditioning | enabled |
| Audio chunk size in checkpoint config | 2048 |
| Audio chunk overlap in checkpoint config | 0 |

Parameter split for the instantiated architecture:

| Component | Parameters |
|---|---:|
| Acoustic subsampling | 2,105,344 |
| Conformer encoder stack | 40,755,456 |
| Cross-attention decoder | 40,027,401 |
| CTC decoder | 6,299,393 |
| Positional encoding | 99,520 |

Important nested folders:

| Folder | Location | Purpose |
|---|---|---|
| `enc_dec_beam_tedlium_dev` | `enc_dec_v2/` | Historical `enc_dec_v2` TEDLIUM dev beam-search decode sweep used to choose `beam=5`, `length_penalty=0.5`. |
| `enc_dec_dynamic_eval` | `enc_dec_v2/` | Historical `enc_dec_v2` teacher-CE dynamic-eval sweep with beam5/lp0.5 teacher and final decoding. |
| `enc_dec_teacher_kl*` | `enc_dec_v2/` | Historical `enc_dec_v2` teacher-KL sweeps, including entropy-filter and relaxed-filter variants. |
| `enc_dec_teacher_epoch_relabel` | `enc_dec_v2/` | Historical teacher relabel ablation. |
| `rob63_earnings_unadapted_sanity` | `enc_dec_v2/` | Earnings22 unadapted sanity check comparing `old_seed` with the older `enc_dec_v2` checkpoint family. |
| `README.md`, `OUTCOME.md`, `summary.csv` | `old_seed/` | Generated ROB-63 checkpoint-specific view for the `old_seed` rows from paired seed-vs-RL artifacts. |
| `rob61_checkpoint_benchmark` | `rl_step_30000/` | Normal/unadapted baselines for `old_seed`, RL checkpoints, and the ROB-63 comparison deltas. |
| `rob63_rl_self_training_compare` | `rl_step_30000/` | Main ROB-63 paired seed-vs-RL one-epoch self-training comparison. Use `rl_step_30000/rob63_rl_self_training_compare/COMBINED_OUTCOME.md` for the full paired readout. |
| `rob63_*_followup` | `rl_step_30000/` | ROB-63 auxiliary comparison follow-ups rolled into the combined ROB-63 summary. |

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
python results/enc_dec/enc_dec_v2/enc_dec_beam_tedlium_dev/aggregate.py
python results/enc_dec/enc_dec_v2/enc_dec_dynamic_eval/aggregate.py
python results/enc_dec/enc_dec_v2/enc_dec_teacher_kl/aggregate.py
python results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_entropy_filter/aggregate.py
python results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_relaxed_filters/aggregate.py
python results/enc_dec/enc_dec_v2/enc_dec_teacher_epoch_relabel/aggregate.py
python results/enc_dec/rl_step_30000/rob61_checkpoint_benchmark/aggregate.py --csv results/enc_dec/rl_step_30000/rob61_checkpoint_benchmark/summary.csv --outcome results/enc_dec/rl_step_30000/rob61_checkpoint_benchmark/OUTCOME.md
python results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py
python results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/pkl --csv results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/summary.csv --outcome results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/OUTCOME.md
python results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/pkl --csv results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/summary.csv --outcome results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/OUTCOME.md
python results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rl_step_30000/rob63_aug_followup/pkl --csv results/enc_dec/rl_step_30000/rob63_aug_followup/summary.csv --outcome results/enc_dec/rl_step_30000/rob63_aug_followup/OUTCOME.md
python results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rl_step_30000/rob63_strong_aug_filter_followup/pkl --csv results/enc_dec/rl_step_30000/rob63_strong_aug_filter_followup/summary.csv --outcome results/enc_dec/rl_step_30000/rob63_strong_aug_filter_followup/OUTCOME.md
python results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/pkl --csv results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/summary.csv --outcome results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/OUTCOME.md
python results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/pkl --extra-directory results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/pkl --extra-directory results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/pkl --extra-directory results/enc_dec/rl_step_30000/rob63_aug_followup/pkl --extra-directory results/enc_dec/rl_step_30000/rob63_strong_aug_filter_followup/pkl --extra-directory results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/pkl --csv results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/combined_summary.csv --outcome results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/COMBINED_OUTCOME.md --checkpoint-view-root results/enc_dec --top-level-outcome results/enc_dec/OUTCOME.md
```

To refresh CSV summaries:

```bash
python results/enc_dec/enc_dec_v2/enc_dec_dynamic_eval/aggregate.py --csv results/enc_dec/enc_dec_v2/enc_dec_dynamic_eval/summary.csv
python results/enc_dec/enc_dec_v2/enc_dec_teacher_kl/aggregate.py --csv results/enc_dec/enc_dec_v2/enc_dec_teacher_kl/summary.csv
python results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_entropy_filter/aggregate.py --csv results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_entropy_filter/summary.csv
python results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_relaxed_filters/aggregate.py --csv results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_relaxed_filters/summary.csv
python results/enc_dec/enc_dec_v2/enc_dec_teacher_epoch_relabel/aggregate.py --csv results/enc_dec/enc_dec_v2/enc_dec_teacher_epoch_relabel/summary.csv
python results/enc_dec/rl_step_30000/rob61_checkpoint_benchmark/aggregate.py --csv results/enc_dec/rl_step_30000/rob61_checkpoint_benchmark/summary.csv --outcome results/enc_dec/rl_step_30000/rob61_checkpoint_benchmark/OUTCOME.md
python results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py --csv results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/summary.csv
python results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/pkl --csv results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/summary.csv --outcome results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/OUTCOME.md
python results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/pkl --csv results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/summary.csv --outcome results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/OUTCOME.md
python results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rl_step_30000/rob63_aug_followup/pkl --csv results/enc_dec/rl_step_30000/rob63_aug_followup/summary.csv --outcome results/enc_dec/rl_step_30000/rob63_aug_followup/OUTCOME.md
python results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rl_step_30000/rob63_strong_aug_filter_followup/pkl --csv results/enc_dec/rl_step_30000/rob63_strong_aug_filter_followup/summary.csv --outcome results/enc_dec/rl_step_30000/rob63_strong_aug_filter_followup/OUTCOME.md
python results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/pkl --csv results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/summary.csv --outcome results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/OUTCOME.md
python results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/pkl --extra-directory results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/pkl --extra-directory results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/pkl --extra-directory results/enc_dec/rl_step_30000/rob63_aug_followup/pkl --extra-directory results/enc_dec/rl_step_30000/rob63_strong_aug_filter_followup/pkl --extra-directory results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/pkl --csv results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/combined_summary.csv --outcome results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/COMBINED_OUTCOME.md --checkpoint-view-root results/enc_dec --top-level-outcome results/enc_dec/OUTCOME.md
```

## Launch Notes

Launch commands inside the child READMEs now point at nested
`results/enc_dec/<checkpoint-family>/...` output folders. For new runs, keep
`RESULTS_DIR` under the relevant checkpoint folder when the launcher supports
it.
