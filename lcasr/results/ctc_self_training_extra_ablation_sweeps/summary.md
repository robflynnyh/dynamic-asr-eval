# CTC Self-Training Extra Ablation Sweep Summary

Generated from `/exp/exp4/acp21rjf/dynamic-asr-eval/lcasr/results/ctc_self_training_extra_ablation_sweeps`.

Focused view: **9e-5 only**, averaged by setting. Per-repeat rows remain in `summary.csv`; grouped metrics are in `summary_by_setting.csv`.

## train_only, 9e-5

| WER | Setting |
|---:|---|
| 5.93% | train-all |
| 6.14% | train-layer-5-only |
| 6.14% | train-layer-1-only |
| 6.15% | train-layer-2-only |
| 6.16% | train-layer-3-only |
| 6.16% | train-layer-4-only |
| 6.21% | train-layer-0-only |
| 6.33% | train-subsampling-only |
| 6.62% | train-ctc-decoder-only |
| 15.75% | train-all |
| 17.46% | train-layer-4-only |
| 17.50% | train-layer-2-only |
| 17.52% | train-layer-3-only |
| 17.58% | train-layer-5-only |
| 17.68% | train-layer-1-only |
| 17.89% | train-layer-0-only |
| 18.54% | train-subsampling-only |
| 19.64% | train-ctc-decoder-only |

## progressive_top, 9e-5

| WER | Setting |
|---:|---|
| 5.90% | freeze-through-1 |
| 5.93% | train-all |
| 5.94% | freeze-subsampling |
| 5.96% | freeze-through-0 |
| 6.00% | freeze-through-2 |
| 6.06% | freeze-through-3 |
| 6.07% | freeze-through-4 |
| 6.65% | freeze-through-5 |
| 15.92% | train-all |
| 15.97% | freeze-through-0 |
| 15.99% | freeze-subsampling |
| 16.11% | freeze-through-1 |
| 16.45% | freeze-through-2 |
| 16.85% | freeze-through-3 |
| 17.24% | freeze-through-4 |
| 19.46% | freeze-through-5 |

## layer_type, 9e-5

| WER | Setting |
|---:|---|
| 6.03% | train-convolution-only |
| 6.12% | train-feed-forward-only |
| 6.15% | train-attention-only |
| 16.49% | train-feed-forward-only |
| 16.87% | train-convolution-only |
| 17.04% | train-attention-only |

## layer_drop_lr_sweep, 9e-5

| WER | Setting |
|---:|---|
| 15.88% | drop-layer-2 |
| 15.89% | drop-subsampling |
| 15.89% | drop-layer-3 |
| 15.91% | drop-layer-5 |
| 15.93% | drop-layer-4 |
| 15.96% | drop-none |
| 16.02% | drop-layer-0 |
| 16.03% | drop-layer-1 |
| 16.11% | drop-ctc-decoder |

