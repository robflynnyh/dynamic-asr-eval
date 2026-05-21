# ROB-67 65536-Context CTC Investigation

This directory groups the ROB-67 65536-context CTC evaluation artifacts under
one result root. Each subdirectory keeps the original PKLs, summaries, and run
notes for a specific comparison.

## Sub-runs

- `unadapted_baseline/`: no-adapt 65536-context baseline at overlap `57344`.
- `self_training_lr1e5/`: adapted 1- and 5-epoch rows at `lr=1e-5`.
- `self_training_higher_lr/`: 1- and 5-epoch follow-up rows at `lr in {9e-5, 3e-4}`.
- `self_training_longer_epochs/`: 10- and 20-epoch follow-up rows at `lr in {9e-5, 3e-4}`.
- `self_training_stride2048/`: final 5-epoch `lr=9e-5` run with overlap `63488`, matching the 16384 setup's stride of `2048`.

## Aggregation

```bash
python lcasr/results/seq_65536_investigation/aggregate_unadapted.py
python lcasr/results/seq_65536_investigation/aggregate_adapted.py
python lcasr/results/seq_65536_investigation/aggregate_adapted.py \
  --root lcasr/results/seq_65536_investigation/self_training_higher_lr
python lcasr/results/seq_65536_investigation/aggregate_adapted.py \
  --root lcasr/results/seq_65536_investigation/self_training_longer_epochs
python lcasr/results/seq_65536_investigation/aggregate_adapted.py \
  --root lcasr/results/seq_65536_investigation/self_training_stride2048
```
