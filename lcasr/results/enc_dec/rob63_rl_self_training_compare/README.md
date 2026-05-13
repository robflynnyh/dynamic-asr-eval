# ROB-63 RL self-training comparison

One-epoch encoder-decoder self-training comparison between the old seed
checkpoint and the ROB-61/ROB-26 RL `step_30000` checkpoint.

## Setup

- Runner: `enc_dec_dynamic_eval_test.py`
- Launcher: `launch_scripts/run_rob63_rl_self_training_compare.sh`
- Datasets: `tedlium`, `earnings22`
- Split: `test`
- Epochs / repeats: `1 / 1`
- Sequence length / overlap: `2048 / 0`
- Decode setting: `beam5_lp0p5`
- Training modes: `teacher_ce`, `teacher_kl`
- Learning rates: `1e-7`, `3e-7`
- Augmentations: `freq6_width34_time0`, `freq3_width24_time0`
- Teacher filters: none
- `old_seed` checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`
- `rl_step_30000` checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt`

This grid is intentionally limited to frequency-mask augmentation and two
conservative learning rates because the issue asks for the checkpoint
comparison rather than a broad hyperparameter search.

The completed TEDLIUM/Earnings22 sweep gives the strongest encoder-decoder
self-training result recorded in this repository so far: RL `step_30000` with
`teacher_ce`, `lr=1e-7`, `freq3_width24_time0`, and beam5/lp0.5 reaches
`7.99%` WER on TEDLIUM and `21.37%` WER on Earnings22. Follow-up passes for
the remaining runner datasets and later CHiME-6/Rev16 diagnostics live in
`results/enc_dec/rob63_best_ce_remaining_datasets/`,
`results/enc_dec/rob63_lower_lr_dev_followup/`, and
`results/enc_dec/rob63_aug_followup/`. Use `COMBINED_OUTCOME.md` as the
single reader-facing summary because it keeps split, unadapted WER, adapted
WER, and checkpoint deltas together.

## Running

Dry run:

```bash
DRY_RUN=1 bash launch_scripts/run_rob63_rl_self_training_compare.sh
```

Queued Mimas launch:

```bash
screen -L -Logfile lcasr/results/enc_dec/rob63_rl_self_training_compare/screen.log \
  -dmS rob63_rl_self_training \
  bash -lc '/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob63_rl_self_training_compare_queued.sh'
```

## Outputs

- Pickles: `results/enc_dec/rob63_rl_self_training_compare/pkl/`
- Logs: `results/enc_dec/rob63_rl_self_training_compare/logs/`
- Summary CSV: `results/enc_dec/rob63_rl_self_training_compare/summary.csv`
- Markdown outcome: `results/enc_dec/rob63_rl_self_training_compare/OUTCOME.md`
- Combined ROB-63 readout: `results/enc_dec/rob63_rl_self_training_compare/COMBINED_OUTCOME.md`

Aggregation:

```bash
python3.10 results/enc_dec/rob63_rl_self_training_compare/aggregate.py \
  --csv results/enc_dec/rob63_rl_self_training_compare/summary.csv \
  --outcome results/enc_dec/rob63_rl_self_training_compare/OUTCOME.md
```

Combined TEDLIUM/Earnings22 grid plus the CHiME-6/Rev16 follow-ups:

```bash
python3.10 results/enc_dec/rob63_rl_self_training_compare/aggregate.py \
  --directory results/enc_dec/rob63_rl_self_training_compare/pkl \
  --extra-directory results/enc_dec/rob63_best_ce_remaining_datasets/pkl \
  --extra-directory results/enc_dec/rob63_lower_lr_dev_followup/pkl \
  --extra-directory results/enc_dec/rob63_aug_followup/pkl \
  --csv results/enc_dec/rob63_rl_self_training_compare/combined_summary.csv \
  --outcome results/enc_dec/rob63_rl_self_training_compare/COMBINED_OUTCOME.md
```

The aggregator also reads the normal decoding benchmark at
`results/enc_dec/rob61_checkpoint_benchmark/summary.csv` by default. It records
the absolute and relative self-training change against the matching normal
decoding row for each checkpoint, dataset, split, and decode setting.
The generated `summary.csv`, `OUTCOME.md`, and `COMBINED_OUTCOME.md` include
the full checkpoint path for each ROB-63 checkpoint label.
