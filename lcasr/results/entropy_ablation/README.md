# ROB-57 Entropy Ablation

This directory contains the ROB-57 ablation for tracing CTC posterior entropy during test-time adaptation.

The approved plan is to run TED-LIUM test and Earnings22 test with 5 adaptation epochs in two settings:

- `freq_mask`: pseudo-label self-training with standard frequency masking, `spec_augment_n_freq_masks=6`, `spec_augment_freq_mask_param=34`, `spec_augment_n_time_masks=0`.
- `no_aug`: pseudo-label self-training with augmentation disabled, `spec_augment_n_freq_masks=0`, `spec_augment_freq_mask_param=0`, `spec_augment_n_time_masks=0`.

Each trace row records clean-chunk mean frame-level CTC posterior entropy before and after optimizer updates. The plotting view uses `pre_update` step 0 plus `post_update` rows so the x-axis is the number of completed adaptation updates.

## Commands

From `lcasr/`:

```bash
EPOCH=5 DATASETS="tedlium earnings22" SETTINGS="freq_mask no_aug" \
  bash launch_scripts/tune_entropy_ablation.sh
```

One-recording smoke test:

```bash
EPOCH=5 DATASETS="tedlium" SETTINGS="freq_mask" MAX_RECORDS=1 \
  bash launch_scripts/tune_entropy_ablation.sh
```

Aggregate and plot from existing raw traces:

```bash
python results/entropy_ablation/aggregate_entropy.py
python results/entropy_ablation/plot_entropy.py --refresh
```

## Outputs

- `raw/*.jsonl`: per-update entropy traces.
- `trace_rows.csv`: flattened trace rows.
- `entropy_by_update.csv`: mean entropy aggregated by dataset, setting, measurement, and update step.
- `entropy_by_update.pdf` and `entropy_by_update.png`: x-axis is completed adaptation updates; y-axis is mean frame entropy.
- `pkl/*.pkl`: normal evaluation artifacts saved by `run_dynamic_eval_full.py`.
- `logs/*.log`: per-run logs.
