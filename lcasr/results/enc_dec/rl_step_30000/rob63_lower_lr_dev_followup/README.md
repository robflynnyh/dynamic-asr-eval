# ROB-63 lower-LR dev follow-up

Follow-up to the ROB-63 CHiME-6/Rev16 result review after the first remaining-dataset
CE pass degraded relative to unadapted decoding.

## Setup

- Runner: `enc_dec_dynamic_eval_test.py`
- Launcher: `launch_scripts/run_rob63_lower_lr_dev_followup.sh`
- Queued wrapper: `scripts/run_rob63_lower_lr_dev_followup_queued.sh`
- Checkpoints: old seed `step_210720.pt` versus RL `step_30000.pt`
- Decode setting: `beam5_lp0p5`
- Training mode: `teacher_ce`
- Learning rates: `3e-8`, `1e-8`, `3e-9`
- Epochs / repeats: `1 / 1`
- Sequence length / overlap: `2048 / 0`
- Augmentation: `freq3_width24_time0`
- Teacher filters: none

## Split policy

- `chime6` runs on `dev`, as requested.
- `rev16` runs on `test` because `enc_dec_dynamic_eval_test.py` asserts that
  Rev16 only supports the test split. This is an explicit runner constraint, not
  a methodological substitution hidden in the output.
- The queued wrapper also runs missing CHiME-6 dev normal-decoding baselines for
  old seed and RL `step_30000` so lower-LR CHiME-6 dev rows can be compared with
  matching unadapted WER.

## Running

Dry run:

```bash
DRY_RUN=1 bash launch_scripts/run_rob63_lower_lr_dev_followup.sh
```

Queued Mimas launch:

```bash
screen -L -Logfile lcasr/results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/screen.log \
  -dmS rob63_lower_lr_dev_followup \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-63 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob63_lower_lr_dev_followup_queued.sh'
```

## Outputs

- Pickles: `results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/pkl/`
- Logs: `results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/logs/`
- Summary CSV: `results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/summary.csv`
- Markdown outcome: `results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/OUTCOME.md`

Aggregation:

```bash
python3.10 results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py \
  --directory results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/pkl \
  --csv results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/summary.csv \
  --outcome results/enc_dec/rl_step_30000/rob63_lower_lr_dev_followup/OUTCOME.md
```
