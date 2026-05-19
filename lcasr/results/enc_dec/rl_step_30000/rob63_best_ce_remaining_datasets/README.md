# ROB-63 best-CE remaining-dataset pass

Follow-up to `rob63_rl_self_training_compare` for thesis-facing coverage on
the remaining encoder-decoder runner datasets.

## Setup

- Runner: `enc_dec_dynamic_eval_test.py`
- Launcher: `launch_scripts/run_rob63_best_ce_remaining_datasets.sh`
- Datasets: `chime6`, `rev16`
- Split: `test`
- Epochs / repeats: `1 / 1`
- Sequence length / overlap: `2048 / 0`
- Decode setting: `beam5_lp0p5`
- Training mode: `teacher_ce`
- Learning rate: `1e-7`
- Augmentation: `freq3_width24_time0`
- Teacher filters: none
- Old seed checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`
- RL checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt`

This uses the best completed ROB-63 CE setting from the TEDLIUM/Earnings22
sweep: CE self-training, `lr=1e-7`, `spec_augment_freq_mask_param=24`,
`spec_augment_n_freq_masks=3`, and `spec_augment_n_time_masks=0`.

## Running

Dry run:

```bash
DRY_RUN=1 bash launch_scripts/run_rob63_best_ce_remaining_datasets.sh
```

Queued Mimas launch:

```bash
screen -L -Logfile lcasr/results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/screen.log \
  -dmS rob63_best_ce_remaining_datasets \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-63 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob63_best_ce_remaining_datasets_queued.sh'
```

## Outputs

- Pickles: `results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/pkl/`
- Logs: `results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/logs/`
- Summary CSV: `results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/summary.csv`
- Markdown outcome: `results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/OUTCOME.md`

Aggregation:

```bash
python3.10 results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py \
  --directory results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/pkl \
  --csv results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/summary.csv \
  --outcome results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/OUTCOME.md
```
