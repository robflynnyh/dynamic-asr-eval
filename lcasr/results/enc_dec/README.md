# Encoder-decoder results

This folder groups the encoder-decoder decoding and test-time adaptation result
sets.

## Layout

| Folder | Purpose |
|---|---|
| `checkpoint1` | Checkpoint-specific ROB-63 view for the normal encoder-decoder seed checkpoint (`old_seed`). This is generated from the ROB-63 combined artifacts and includes `summary.csv` plus a focused `OUTCOME.md`. |
| `checkpoint2` | Checkpoint-specific ROB-63 view for the newest 30K RL checkpoint (`rl_step_30000`). This is generated from the same ROB-63 combined artifacts and includes `summary.csv` plus a focused `OUTCOME.md`. |
| `enc_dec_beam_tedlium_dev` | TEDLIUM dev beam-search decode sweep used to choose `beam=5`, `length_penalty=0.5` as the default quality/runtime decode. |
| `enc_dec_dynamic_eval` | Teacher-CE dynamic-eval sweep with beam5/lp0.5 teacher and final decoding, including same-decode no-adapt baselines. |
| `enc_dec_teacher_kl` | Teacher-KL dynamic-eval sweep with the original teacher filters. |
| `enc_dec_teacher_kl_entropy_filter` | Teacher-KL sweep with additional low-confidence entropy filtering. |
| `enc_dec_teacher_kl_relaxed_filters` | Teacher-KL sweep with relaxed repeated-token filtering and no CTC agreement filter. |
| `enc_dec_teacher_epoch_relabel` | Opt-in CE/KL teacher relabel ablation where each epoch labels and filters all chunks before student training. |
| `rob63_rl_self_training_compare` | ROB-63 seed vs RL `step_30000` encoder-decoder self-training comparison without teacher filters; use `COMBINED_OUTCOME.md` here as the single readout for normal/unadapted WER and adapted WER. |
| `rob63_best_ce_remaining_datasets` | Auxiliary ROB-63 best-CE CHiME-6/Rev16 follow-up whose pickles are rolled into `rob63_rl_self_training_compare/COMBINED_OUTCOME.md`. |
| `rob63_lower_lr_dev_followup` | Auxiliary ROB-63 lower-LR CHiME-6 dev / Rev16 test CE follow-up after the first remaining-dataset pass degraded versus normal decoding. |
| `rob63_aug_followup` | Auxiliary ROB-63 CHiME-6 dev / Rev16 test CE follow-up that tries no augmentation and lighter frequency masking after the lower-LR pass remained deletion-heavy. |
| `rob63_strong_aug_filter_followup` | Auxiliary ROB-63 stronger-frequency-mask and light repeat-filter follow-up for CHiME-6 dev and Rev16 test. |
| `rob63_targeted_high_aug_followup` | Auxiliary ROB-63 targeted strong-mask follow-up for Rev16 `1e-7` and Earnings22 `3e-8`. |
| `rob63_earnings_unadapted_sanity` | ROB-63 sanity rerun comparing Earnings22 unadapted WER for `old_seed` against the older `enc_dec/OUTCOME` checkpoint family. |

ROB-63 checkpoint mapping:

| Folder | Checkpoint key | Checkpoint path |
|---|---|---|
| `checkpoint1` | `old_seed` | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt` |
| `checkpoint2` | `rl_step_30000` | `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt` |

The checkpoint folders are derived views, not duplicate raw-artifact stores.
They make the seed-vs-RL comparison easier to read while the original result
folders keep their logs, PKLs, and per-run outcomes.

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

Run any child aggregate from the repo root:

```bash
python results/enc_dec/enc_dec_beam_tedlium_dev/aggregate.py
python results/enc_dec/enc_dec_dynamic_eval/aggregate.py
python results/enc_dec/enc_dec_teacher_kl/aggregate.py
python results/enc_dec/enc_dec_teacher_kl_entropy_filter/aggregate.py
python results/enc_dec/enc_dec_teacher_kl_relaxed_filters/aggregate.py
python results/enc_dec/enc_dec_teacher_epoch_relabel/aggregate.py
python results/enc_dec/rob61_checkpoint_benchmark/aggregate.py --csv results/enc_dec/rob61_checkpoint_benchmark/summary.csv --outcome results/enc_dec/rob61_checkpoint_benchmark/OUTCOME.md
python results/enc_dec/rob63_rl_self_training_compare/aggregate.py
python results/enc_dec/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rob63_best_ce_remaining_datasets/pkl --csv results/enc_dec/rob63_best_ce_remaining_datasets/summary.csv --outcome results/enc_dec/rob63_best_ce_remaining_datasets/OUTCOME.md
python results/enc_dec/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rob63_lower_lr_dev_followup/pkl --csv results/enc_dec/rob63_lower_lr_dev_followup/summary.csv --outcome results/enc_dec/rob63_lower_lr_dev_followup/OUTCOME.md
python results/enc_dec/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rob63_aug_followup/pkl --csv results/enc_dec/rob63_aug_followup/summary.csv --outcome results/enc_dec/rob63_aug_followup/OUTCOME.md
python results/enc_dec/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rob63_strong_aug_filter_followup/pkl --csv results/enc_dec/rob63_strong_aug_filter_followup/summary.csv --outcome results/enc_dec/rob63_strong_aug_filter_followup/OUTCOME.md
python results/enc_dec/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rob63_targeted_high_aug_followup/pkl --csv results/enc_dec/rob63_targeted_high_aug_followup/summary.csv --outcome results/enc_dec/rob63_targeted_high_aug_followup/OUTCOME.md
python results/enc_dec/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rob63_rl_self_training_compare/pkl --extra-directory results/enc_dec/rob63_best_ce_remaining_datasets/pkl --extra-directory results/enc_dec/rob63_lower_lr_dev_followup/pkl --extra-directory results/enc_dec/rob63_aug_followup/pkl --extra-directory results/enc_dec/rob63_strong_aug_filter_followup/pkl --extra-directory results/enc_dec/rob63_targeted_high_aug_followup/pkl --csv results/enc_dec/rob63_rl_self_training_compare/combined_summary.csv --outcome results/enc_dec/rob63_rl_self_training_compare/COMBINED_OUTCOME.md --checkpoint-view-root results/enc_dec --top-level-outcome results/enc_dec/OUTCOME.md
```

To refresh CSV summaries:

```bash
python results/enc_dec/enc_dec_dynamic_eval/aggregate.py --csv results/enc_dec/enc_dec_dynamic_eval/summary.csv
python results/enc_dec/enc_dec_teacher_kl/aggregate.py --csv results/enc_dec/enc_dec_teacher_kl/summary.csv
python results/enc_dec/enc_dec_teacher_kl_entropy_filter/aggregate.py --csv results/enc_dec/enc_dec_teacher_kl_entropy_filter/summary.csv
python results/enc_dec/enc_dec_teacher_kl_relaxed_filters/aggregate.py --csv results/enc_dec/enc_dec_teacher_kl_relaxed_filters/summary.csv
python results/enc_dec/enc_dec_teacher_epoch_relabel/aggregate.py --csv results/enc_dec/enc_dec_teacher_epoch_relabel/summary.csv
python results/enc_dec/rob61_checkpoint_benchmark/aggregate.py --csv results/enc_dec/rob61_checkpoint_benchmark/summary.csv --outcome results/enc_dec/rob61_checkpoint_benchmark/OUTCOME.md
python results/enc_dec/rob63_rl_self_training_compare/aggregate.py --csv results/enc_dec/rob63_rl_self_training_compare/summary.csv
python results/enc_dec/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rob63_best_ce_remaining_datasets/pkl --csv results/enc_dec/rob63_best_ce_remaining_datasets/summary.csv --outcome results/enc_dec/rob63_best_ce_remaining_datasets/OUTCOME.md
python results/enc_dec/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rob63_lower_lr_dev_followup/pkl --csv results/enc_dec/rob63_lower_lr_dev_followup/summary.csv --outcome results/enc_dec/rob63_lower_lr_dev_followup/OUTCOME.md
python results/enc_dec/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rob63_aug_followup/pkl --csv results/enc_dec/rob63_aug_followup/summary.csv --outcome results/enc_dec/rob63_aug_followup/OUTCOME.md
python results/enc_dec/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rob63_strong_aug_filter_followup/pkl --csv results/enc_dec/rob63_strong_aug_filter_followup/summary.csv --outcome results/enc_dec/rob63_strong_aug_filter_followup/OUTCOME.md
python results/enc_dec/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rob63_targeted_high_aug_followup/pkl --csv results/enc_dec/rob63_targeted_high_aug_followup/summary.csv --outcome results/enc_dec/rob63_targeted_high_aug_followup/OUTCOME.md
python results/enc_dec/rob63_rl_self_training_compare/aggregate.py --directory results/enc_dec/rob63_rl_self_training_compare/pkl --extra-directory results/enc_dec/rob63_best_ce_remaining_datasets/pkl --extra-directory results/enc_dec/rob63_lower_lr_dev_followup/pkl --extra-directory results/enc_dec/rob63_aug_followup/pkl --extra-directory results/enc_dec/rob63_strong_aug_filter_followup/pkl --extra-directory results/enc_dec/rob63_targeted_high_aug_followup/pkl --csv results/enc_dec/rob63_rl_self_training_compare/combined_summary.csv --outcome results/enc_dec/rob63_rl_self_training_compare/COMBINED_OUTCOME.md --checkpoint-view-root results/enc_dec --top-level-outcome results/enc_dec/OUTCOME.md
```

## Launch Notes

Launch commands inside the child READMEs now point at nested
`results/enc_dec/...` output folders. For new runs, keep `RESULTS_DIR` under
this parent folder when the launcher supports it.
