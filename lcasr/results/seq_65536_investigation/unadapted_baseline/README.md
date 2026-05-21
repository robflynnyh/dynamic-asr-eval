# CTC 65536-Context Unadapted Baseline

ROB-67 no-adapt baseline for the 65536-context CTC checkpoint. This is the unadapted comparison for the adapted 65536-context self-training eval.

- Mimas checkpoint: `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_65536_rp_1/step_105360.pt`
- Stanage checkpoint: `/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_65536_rp_1/step_105360.pt`
- Datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Split: `test`
- Sequence length: `65536`
- Overlap: `57344`
- Adaptation epochs: `0`
- Decode: greedy CTC
- Repeats: `1`

Manual launch from `lcasr/`:

```bash
DATASETS="earnings22 tedlium chime6 rev16" REPEATS=1 \
  bash launch_scripts/run_ctc_seq65536_unadapted_baseline.sh
```

Aggregate after completion:

```bash
python lcasr/results/seq_65536_investigation/aggregate_unadapted.py
```

Investigate against shorter exact no-adapt baselines:

```bash
python lcasr/results/seq_65536_investigation/investigate_unadapted_baseline.py
```

The investigation report is
`lcasr/results/seq_65536_investigation/UNADAPTED_BASELINE_INVESTIGATION.md`.
