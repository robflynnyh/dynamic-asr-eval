# ROB-63 stronger-augmentation/filter follow-up

Follow-up to the human clarification on 2026-05-13 that the ROB-63
augmentation check should try stronger augmentation rather than weaker/no
augmentation, with a small teacher-filtering check.

## Setup

- Runner: `enc_dec_dynamic_eval_test.py`
- Launcher: `launch_scripts/run_rob63_strong_aug_filter_followup.sh`
- Queued wrapper: `scripts/run_rob63_strong_aug_filter_followup_queued.sh`
- Checkpoints: old seed `step_210720.pt` versus RL `step_30000.pt`
- Decode setting: `beam5_lp0p5`
- Training mode: `teacher_ce`
- Learning rates: `1e-8`, `3e-8`
- Epochs / repeats: `1 / 1`
- Sequence length / overlap: `2048 / 0`
- Augmentation: `freq9_width44_time0`
- Teacher filter settings: `no_filter`, `basic_repeat_filter`

`basic_repeat_filter` keeps the filter light: maximum teacher length, maximum
consecutive token repeat, and maximum consecutive word repeat. It avoids
confidence and decode-agreement filters so the follow-up stays focused on
obvious faulty repeated teacher predictions.

## Split policy

- `chime6` runs on `dev` to match the lower-LR/lighter-augmentation follow-ups
  and their normal-decoding baselines.
- `rev16` runs on `test` because `enc_dec_dynamic_eval_test.py` asserts that
  Rev16 only supports the test split.

## Running

Dry run:

```bash
DRY_RUN=1 bash launch_scripts/run_rob63_strong_aug_filter_followup.sh
```

Queued Mimas launch:

```bash
screen -L -Logfile lcasr/results/enc_dec/rob63_strong_aug_filter_followup/screen.log \
  -dmS rob63_strong_aug_filter_followup \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-63 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob63_strong_aug_filter_followup_queued.sh'
```

## Outputs

- Pickles: `results/enc_dec/rob63_strong_aug_filter_followup/pkl/`
- Logs: `results/enc_dec/rob63_strong_aug_filter_followup/logs/`
- Summary CSV: `results/enc_dec/rob63_strong_aug_filter_followup/summary.csv`
- Markdown outcome: `results/enc_dec/rob63_strong_aug_filter_followup/OUTCOME.md`

Aggregation:

```bash
python3.10 results/enc_dec/rob63_rl_self_training_compare/aggregate.py \
  --directory results/enc_dec/rob63_strong_aug_filter_followup/pkl \
  --csv results/enc_dec/rob63_strong_aug_filter_followup/summary.csv \
  --outcome results/enc_dec/rob63_strong_aug_filter_followup/OUTCOME.md
```
