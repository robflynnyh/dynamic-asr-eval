# ROB-63 augmentation follow-up

Follow-up to the lower-LR CHiME-6/Rev16 pass after `freq3_width24_time0`
remained deletion-dominated on CHiME-6 dev. This run keeps the comparison narrow
and tests whether the degradation is driven by the frequency-mask setting rather
than by the seed/RL checkpoint comparison itself.

## Setup

- Runner: `enc_dec_dynamic_eval_test.py`
- Launcher: `launch_scripts/run_rob63_aug_followup.sh`
- Queued wrapper: `scripts/run_rob63_aug_followup_queued.sh`
- Checkpoints: old seed `step_210720.pt` versus RL `step_30000.pt`
- Decode setting: `beam5_lp0p5`
- Training mode: `teacher_ce`
- Learning rates: `1e-8`, `3e-8`
- Epochs / repeats: `1 / 1`
- Sequence length / overlap: `2048 / 0`
- Augmentations: `no_aug`, `freq1_width12_time0`
- Teacher filters: none

## Split policy

- `chime6` runs on `dev` to match the lower-LR follow-up and its matching
  normal-decoding baselines.
- `rev16` runs on `test` because `enc_dec_dynamic_eval_test.py` asserts that
  Rev16 only supports the test split.

## Running

Dry run:

```bash
DRY_RUN=1 bash launch_scripts/run_rob63_aug_followup.sh
```

Queued Mimas launch:

```bash
screen -L -Logfile lcasr/results/enc_dec/rob63_aug_followup/screen.log \
  -dmS rob63_aug_followup \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-63 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob63_aug_followup_queued.sh'
```

## Outputs

- Pickles: `results/enc_dec/rob63_aug_followup/pkl/`
- Logs: `results/enc_dec/rob63_aug_followup/logs/`
- Summary CSV: `results/enc_dec/rob63_aug_followup/summary.csv`
- Markdown outcome: `results/enc_dec/rob63_aug_followup/OUTCOME.md`

The completed run did not rescue CHiME-6 dev: all adapted CHiME-6 cells stay at
`1.00000` WER for both checkpoints, worse than the matching unadapted baselines
(`0.83439` old seed, `0.81157` RL). On Rev16 test, the only improved
checkpoint-comparison cell is RL `step_30000` with `lr=3e-8` and
`freq1_width12_time0`, at `0.19263` WER versus the matching old-seed adapted
cell at `0.24183`; it still remains above the RL unadapted WER of `0.17206`.

Aggregation:

```bash
python3.10 results/enc_dec/rob63_rl_self_training_compare/aggregate.py \
  --directory results/enc_dec/rob63_aug_followup/pkl \
  --csv results/enc_dec/rob63_aug_followup/summary.csv \
  --outcome results/enc_dec/rob63_aug_followup/OUTCOME.md
```
