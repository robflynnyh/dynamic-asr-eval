# CTC 8192-Context Self-Training Summary

Generated from `/exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-115/lcasr/results/ctc_seq8192_self_training_eval`.

Per-repeat rows: `8`.
Grouped rows: `8`.

| Dataset | Epochs | LR | N | WER Mean | WER Std | Ins | Del | Sub | Words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chime6 | 5 | 6em5 | 1 | 60.08% |  | 1.03% | 40.33% | 18.72% | 56740 |
| chime6 | 5 | 9em5 | 1 | 59.52% |  | 1.14% | 37.45% | 20.93% | 56740 |
| earnings22 | 5 | 6em5 | 1 | 15.18% |  | 2.62% | 3.37% | 9.18% | 48963 |
| earnings22 | 5 | 9em5 | 1 | 15.23% |  | 2.62% | 3.41% | 9.20% | 48963 |
| rev16 | 5 | 6em5 | 1 | 14.32% |  | 2.54% | 4.85% | 6.93% | 195833 |
| rev16 | 5 | 9em5 | 1 | 14.49% |  | 2.56% | 4.92% | 7.01% | 195833 |
| tedlium | 5 | 6em5 | 1 | 5.76% |  | 0.85% | 1.69% | 3.23% | 28215 |
| tedlium | 5 | 9em5 | 1 | 5.85% |  | 0.83% | 1.71% | 3.31% | 28215 |

## ROB-110 8192 No-Adapt Baseline

Imported from `/exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-115/lcasr/results/ctc_seq8192_unadapted_baseline/summary.csv`.

| Dataset | Split | Epochs | N | WER Mean | Ins | Del | Sub | Words |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| chime6 | test | 0 | 1 | 84.34% | 0.12% | 81.20% | 3.01% | 56740 |
| earnings22 | test | 0 | 1 | 18.68% | 2.32% | 5.23% | 11.13% | 48963 |
| rev16 | test | 0 | 1 | 15.17% | 1.72% | 6.97% | 6.47% | 195833 |
| tedlium | test | 0 | 1 | 6.31% | 0.74% | 2.09% | 3.47% | 28215 |

## Adapted Delta From 8192 No-Adapt

| Dataset | LR | Adapted WER | No-Adapt WER | Absolute Delta | Relative Delta |
|---|---:|---:|---:|---:|---:|
| chime6 | 6em5 | 60.08% | 84.34% | -24.25% | -28.76% |
| chime6 | 9em5 | 59.52% | 84.34% | -24.82% | -29.43% |
| earnings22 | 6em5 | 15.18% | 18.68% | -3.50% | -18.73% |
| earnings22 | 9em5 | 15.23% | 18.68% | -3.45% | -18.46% |
| rev16 | 6em5 | 14.32% | 15.17% | -0.85% | -5.57% |
| rev16 | 9em5 | 14.49% | 15.17% | -0.68% | -4.46% |
| tedlium | 6em5 | 5.76% | 6.31% | -0.55% | -8.66% |
| tedlium | 9em5 | 5.85% | 6.31% | -0.46% | -7.25% |
