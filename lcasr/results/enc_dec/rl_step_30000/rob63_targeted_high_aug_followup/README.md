# ROB-63 targeted high-augmentation follow-up

Follow-up to the 2026-05-13 human request for two targeted cells after the
stronger Rev16 augmentation result:

1. Rev16 with the stronger augmentation and the higher `1e-7` learning rate.
2. Earnings22 with the stronger augmentation and `3e-8`.

## Setup

- Runner: `enc_dec_dynamic_eval_test.py`
- Launcher: `launch_scripts/run_rob63_targeted_high_aug_followup.sh`
- Queued wrapper: `scripts/run_rob63_targeted_high_aug_followup_queued.sh`
- Checkpoints: `old_seed` versus `rl_step_30000`
- Decode setting: `beam5_lp0p5`
- Training mode: `teacher_ce`
- Rev16 learning rate: `1e-7`
- Earnings22 learning rate: `3e-8`
- Epochs / repeats: `1 / 1`
- Sequence length / overlap: `2048 / 0`
- Augmentation: `freq9_width44_time0`
- Teacher filter setting: `no_filter`

Checkpoint key:

- `old_seed`: normal encoder-decoder seed checkpoint at `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`.
- `rl_step_30000`: 30K RL-trained checkpoint at `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt`.

## Split Policy

- `rev16` runs on `test` because `enc_dec_dynamic_eval_test.py` asserts that
  Rev16 only supports the test split.
- `earnings22` runs on `test` to match the main ROB-63 TED-LIUM/Earnings grid.

## Running

Dry run:

```bash
DRY_RUN=1 bash launch_scripts/run_rob63_targeted_high_aug_followup.sh
```

Queued Mimas launch:

```bash
screen -L -Logfile lcasr/results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/screen.log \
  -dmS rob63_targeted_high_aug_followup \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-63 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob63_targeted_high_aug_followup_queued.sh'
```

## Outputs

- Pickles: `results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/pkl/`
- Logs: `results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/logs/`
- Summary CSV: `results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/summary.csv`
- Markdown outcome: `results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/OUTCOME.md`

Aggregation:

```bash
python3.10 results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py \
  --directory results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/pkl \
  --csv results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/summary.csv \
  --outcome results/enc_dec/rl_step_30000/rob63_targeted_high_aug_followup/OUTCOME.md
```
