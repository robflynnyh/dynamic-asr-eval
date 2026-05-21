# CTC 8192-Context Unadapted Baseline

ROB-112 no-adapt baseline for the 8192-context CTC checkpoint. This is the follow-up comparison row requested from ROB-67, between the existing 2048-context and 65536-context no-adapt baselines.

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_8192_rp_1/step_105360.pt`
- Datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Split: `test`
- Sequence length: `8192`
- Overlap: `7168`
- Stride: `1024`
- Adaptation epochs: `0`
- Decode: greedy CTC
- Repeats: `1`

Launch from the repo root with the callback-backed wrapper:

```bash
screen -L -Logfile lcasr/results/ctc_seq8192_unadapted_baseline/logs/rob112_ctc_seq8192_unadapted_baseline.screen.log \
  -dmS rob112_ctc_seq8192_unadapted_baseline \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-112 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob112_ctc_seq8192_unadapted_baseline_queued.sh'
```

The wrapper runs the eval and then regenerates:

- `summary.csv`: thesis-friendly dataset-level WER table.
- `summary.md`: compact Markdown WER table.
- `context_baseline_comparison.csv`: joined no-adapt comparison against existing 2048, optional 16384, and 65536 summaries.
- `context_baseline_comparison.md`: compact comparison table for thesis checks.

Manual aggregation command:

```bash
python lcasr/results/ctc_seq8192_unadapted_baseline/aggregate.py
python lcasr/results/ctc_seq8192_unadapted_baseline/compare_context_baselines.py
```

The 16384 row is optional because ROB-110 tracks that result separately. When it is unavailable, the comparison helper records the missing summary path and still writes the 2048/8192/65536 comparison.
