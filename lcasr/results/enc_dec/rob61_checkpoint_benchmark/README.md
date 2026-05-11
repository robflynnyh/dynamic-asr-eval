# ROB-61 checkpoint benchmark

Normal encoder-decoder inference benchmark for the ROB-26 RL checkpoints against
the old encoder-decoder seed checkpoint.

## Setup

- Runner: `enc_dec_inference_test.py`
- Launcher: `launch_scripts/run_rob61_checkpoint_benchmark.sh`
- Datasets: `tedlium`, `earnings22`
- ROB-63 follow-up also adds the beam5/lp0.5 normal baselines for
  `chime6` and `rev16` for `old_seed` and `rl_step_30000`.
- Split: `test`
- Sequence length / overlap: `2048 / 0`
- Decode settings: greedy and `beam5_lp0p5`
- Old checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`
- RL checkpoint root:
  `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5`
- RL steps: `2000`, `10000`, `20000`, `30000`

## Running

```bash
bash launch_scripts/run_rob61_checkpoint_benchmark.sh
```

Dry run:

```bash
DRY_RUN=1 bash launch_scripts/run_rob61_checkpoint_benchmark.sh
```

Queued Mimas launch:

```bash
screen -L -Logfile lcasr/results/enc_dec/rob61_checkpoint_benchmark/screen.log \
  -dmS rob61_checkpoint_benchmark \
  bash -lc '/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob61_checkpoint_benchmark_queued.sh'
```

ROB-63 CHiME-6/Rev16 normal-baseline follow-up:

```bash
DRY_RUN=1 bash launch_scripts/run_rob63_remaining_normal_baselines.sh

screen -L -Logfile lcasr/results/enc_dec/rob61_checkpoint_benchmark/rob63_remaining_normal_baselines_screen.log \
  -dmS rob63_remaining_normal_baselines \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-63 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob63_remaining_normal_baselines_queued.sh'
```

## Outputs

- Pickles: `results/enc_dec/rob61_checkpoint_benchmark/pkl/`
- Logs: `results/enc_dec/rob61_checkpoint_benchmark/logs/`
- Summary CSV: `results/enc_dec/rob61_checkpoint_benchmark/summary.csv`
- Markdown outcome: `results/enc_dec/rob61_checkpoint_benchmark/OUTCOME.md`

Aggregation:

```bash
python3.10 results/enc_dec/rob61_checkpoint_benchmark/aggregate.py \
  --csv results/enc_dec/rob61_checkpoint_benchmark/summary.csv \
  --outcome results/enc_dec/rob61_checkpoint_benchmark/OUTCOME.md
```
