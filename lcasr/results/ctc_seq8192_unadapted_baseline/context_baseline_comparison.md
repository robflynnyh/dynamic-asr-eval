# CTC No-Adapt Context Baseline Comparison

| Dataset | Split | Seq Len | Overlap | Stride | WER | Ins | Del | Sub |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| chime6 | test | 2048 | 1792 | 256 | 85.27% | 0.13% | 82.62% | 2.51% |
| chime6 | test | 8192 | 7168 | 1024 | 84.34% | 0.12% | 81.20% | 3.01% |
| chime6 | test | 65536 | 57344 | 8192 | 86.76% | 0.10% | 84.51% | 2.15% |
| earnings22 | test | 2048 | 1792 | 256 | 19.54% | 2.55% | 5.09% | 11.90% |
| earnings22 | test | 8192 | 7168 | 1024 | 18.68% | 2.32% | 5.23% | 11.13% |
| earnings22 | test | 65536 | 57344 | 8192 | 18.32% | 2.19% | 5.17% | 10.96% |
| rev16 | test | 2048 | 1792 | 256 | 15.27% | 1.80% | 6.87% | 6.60% |
| rev16 | test | 8192 | 7168 | 1024 | 15.17% | 1.72% | 6.97% | 6.47% |
| rev16 | test | 65536 | 57344 | 8192 | 15.21% | 1.70% | 7.13% | 6.38% |
| tedlium | test | 2048 | 1792 | 256 | 6.54% | 0.81% | 2.00% | 3.73% |
| tedlium | test | 8192 | 7168 | 1024 | 6.31% | 0.74% | 2.09% | 3.47% |
| tedlium | test | 65536 | 57344 | 8192 | 6.17% | 0.71% | 2.05% | 3.42% |

## Sources

- loaded 4 rows: /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-112/lcasr/results/ctc_seq2048_unadapted_baseline/summary.csv
- loaded 4 rows: /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-112/lcasr/results/ctc_seq8192_unadapted_baseline/summary.csv
- missing: /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-112/lcasr/results/ctc_seq16384_unadapted_baseline/summary.csv
- loaded 4 rows: /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-112/lcasr/results/seq_65536_investigation/unadapted_baseline/summary.csv
