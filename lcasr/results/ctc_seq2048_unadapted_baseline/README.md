# CTC 2048-Context Unadapted Baseline

ROB-66 no-adapt baseline for the 2048-context CTC checkpoint used by ROB-56.
This is the comparison column needed for the Chapter 7 sequence-length table.

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt`
- Datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Split: `test`
- Sequence length: `2048`
- Overlap: `1792`
- Adaptation epochs: `0`
- Decode: greedy CTC
- Repeats: `1`

Launch from the repo root with the callback-backed wrapper:

```bash
screen -L -Logfile lcasr/results/ctc_seq2048_unadapted_baseline/logs/rob66_ctc_seq2048_unadapted_baseline.screen.log \
  -dmS rob66_ctc_seq2048_unadapted_baseline \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-66 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob66_ctc_seq2048_unadapted_baseline_queued.sh'
```

The wrapper runs the eval and then regenerates:

- `summary.csv`: thesis-friendly dataset-level WER table.
- `summary.md`: compact Markdown WER table.

Manual aggregation command:

```bash
python lcasr/results/ctc_seq2048_unadapted_baseline/aggregate.py
```
