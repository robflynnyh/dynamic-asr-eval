# CTC 16384-Context Unadapted Baseline

ROB-110 no-adapt baseline for the 16384-context CTC checkpoint. This fills the missing 2.7 minute comparison row between the committed ROB-66 2048-context baseline and ROB-67 65536-context baseline.

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_16384_rp_1/step_105360.pt`
- Datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Split: `test`
- Sequence length: `16384`
- Overlap: `14336`
- Stride: `2048`
- Adaptation epochs: `0`
- Decode: greedy CTC
- Repeats: `1`
- Output directory: `lcasr/results/ctc_seq16384_unadapted_baseline/`

Launch from the repo root with the callback-backed wrapper:

```bash
screen -L -Logfile lcasr/results/ctc_seq16384_unadapted_baseline/logs/rob110_ctc_seq16384_unadapted_baseline.screen.log \
  -dmS rob110_ctc_seq16384_unadapted_baseline \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-110 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob110_ctc_seq16384_unadapted_baseline_queued.sh'
```

The wrapper runs the eval and then regenerates:

- `summary.csv`: dataset-level WER table generated from local PKL artifacts.
- `summary.md`: compact Markdown WER table generated from local PKL artifacts.
- `context_baseline_comparison.csv`: regenerated comparison against the existing 2048 and 65536 no-adapt baseline summaries.
- `context_baseline_comparison.md`: compact thesis-facing comparison table.

Manual aggregation after completion:

```bash
python lcasr/results/ctc_seq16384_unadapted_baseline/aggregate.py
python lcasr/results/ctc_seq16384_unadapted_baseline/compare_context_baselines.py
```

Narrow smoke-test shape before the full run:

```bash
cd lcasr
DATASETS=earnings22 MAX_RECORDS=1 RESULTS_DIR=./results/ctc_seq16384_unadapted_baseline/smoke \
  bash launch_scripts/run_ctc_seq16384_unadapted_baseline.sh
```
