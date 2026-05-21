# ROB-115 CTC Context Comparison Snapshot

This is a snapshot, not a final thesis-ready comparison. It is generated
from committed summary artifacts available in this checkout.

| Source | Dataset | Split | Seq Len | Overlap | Epochs | LR | N | WER | Summary |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 2048 adapted | chime6 | test | 2048 | 1792 | 1 | 9em5 | 1 | 66.05% | `lcasr/results/ctc_seq2048_self_training_eval/summary_by_setting.csv` |
| 2048 adapted | chime6 | test | 2048 | 1792 | 5 | 9em5 | 1 | 63.94% | `lcasr/results/ctc_seq2048_self_training_eval/summary_by_setting.csv` |
| 2048 adapted | earnings22 | test | 2048 | 1792 | 1 | 9em5 | 1 | 16.41% | `lcasr/results/ctc_seq2048_self_training_eval/summary_by_setting.csv` |
| 2048 adapted | earnings22 | test | 2048 | 1792 | 5 | 9em5 | 1 | 16.82% | `lcasr/results/ctc_seq2048_self_training_eval/summary_by_setting.csv` |
| 2048 adapted | rev16 | test | 2048 | 1792 | 1 | 9em5 | 1 | 15.01% | `lcasr/results/ctc_seq2048_self_training_eval/summary_by_setting.csv` |
| 2048 adapted | rev16 | test | 2048 | 1792 | 5 | 9em5 | 1 | 53.97% | `lcasr/results/ctc_seq2048_self_training_eval/summary_by_setting.csv` |
| 2048 adapted | tedlium | test | 2048 | 1792 | 1 | 9em5 | 1 | 6.11% | `lcasr/results/ctc_seq2048_self_training_eval/summary_by_setting.csv` |
| 2048 adapted | tedlium | test | 2048 | 1792 | 5 | 9em5 | 1 | 16.57% | `lcasr/results/ctc_seq2048_self_training_eval/summary_by_setting.csv` |
| 2048 no-adapt | chime6 | test | 2048 | 1792 | 0 |  | 1 | 85.27% | `lcasr/results/ctc_seq2048_unadapted_baseline/summary.csv` |
| 2048 no-adapt | earnings22 | test | 2048 | 1792 | 0 |  | 1 | 19.54% | `lcasr/results/ctc_seq2048_unadapted_baseline/summary.csv` |
| 2048 no-adapt | rev16 | test | 2048 | 1792 | 0 |  | 1 | 15.27% | `lcasr/results/ctc_seq2048_unadapted_baseline/summary.csv` |
| 2048 no-adapt | tedlium | test | 2048 | 1792 | 0 |  | 1 | 6.54% | `lcasr/results/ctc_seq2048_unadapted_baseline/summary.csv` |
| 8192 adapted | chime6 | test | 8192 | 7168 | 5 | 9em5 | 1 | 59.52% | `lcasr/results/ctc_seq8192_self_training_eval/summary_by_setting.csv` |
| 8192 adapted | earnings22 | test | 8192 | 7168 | 5 | 9em5 | 1 | 15.23% | `lcasr/results/ctc_seq8192_self_training_eval/summary_by_setting.csv` |
| 8192 adapted | rev16 | test | 8192 | 7168 | 5 | 9em5 | 1 | 14.49% | `lcasr/results/ctc_seq8192_self_training_eval/summary_by_setting.csv` |
| 8192 adapted | tedlium | test | 8192 | 7168 | 5 | 9em5 | 1 | 5.85% | `lcasr/results/ctc_seq8192_self_training_eval/summary_by_setting.csv` |
| 8192 no-adapt | chime6 | test | 8192 | 7168 | 0 |  | 1 | 84.34% | `lcasr/results/ctc_seq8192_unadapted_baseline/summary.csv` |
| 8192 no-adapt | earnings22 | test | 8192 | 7168 | 0 |  | 1 | 18.68% | `lcasr/results/ctc_seq8192_unadapted_baseline/summary.csv` |
| 8192 no-adapt | rev16 | test | 8192 | 7168 | 0 |  | 1 | 15.17% | `lcasr/results/ctc_seq8192_unadapted_baseline/summary.csv` |
| 8192 no-adapt | tedlium | test | 8192 | 7168 | 0 |  | 1 | 6.31% | `lcasr/results/ctc_seq8192_unadapted_baseline/summary.csv` |
| 16384 adapted RMM | chime6 | test | 16384 | 14336 | 1 | 9em5 | 3 | 100.00% | `lcasr/results/rmm_eval/ctc_seq16384/summary_by_setting.csv` |
| 16384 adapted RMM | chime6 | test | 16384 | 14336 | 5 | 9em5 | 3 | 100.00% | `lcasr/results/rmm_eval/ctc_seq16384/summary_by_setting.csv` |
| 16384 adapted RMM | earnings22 | test | 16384 | 14336 | 1 | 9em5 | 3 | 15.84% | `lcasr/results/rmm_eval/ctc_seq16384/summary_by_setting.csv` |
| 16384 adapted RMM | earnings22 | test | 16384 | 14336 | 5 | 9em5 | 3 | 15.35% | `lcasr/results/rmm_eval/ctc_seq16384/summary_by_setting.csv` |
| 16384 adapted RMM | rev16 | test | 16384 | 14336 | 1 | 9em5 | 3 | 14.21% | `lcasr/results/rmm_eval/ctc_seq16384/summary_by_setting.csv` |
| 16384 adapted RMM | rev16 | test | 16384 | 14336 | 5 | 9em5 | 3 | 14.04% | `lcasr/results/rmm_eval/ctc_seq16384/summary_by_setting.csv` |
| 16384 adapted RMM | tedlium | test | 16384 | 14336 | 1 | 9em5 | 3 | 5.97% | `lcasr/results/rmm_eval/ctc_seq16384/summary_by_setting.csv` |
| 16384 adapted RMM | tedlium | test | 16384 | 14336 | 5 | 9em5 | 3 | 5.79% | `lcasr/results/rmm_eval/ctc_seq16384/summary_by_setting.csv` |
| 16384 no-adapt | chime6 | test | 16384 | 14336 | 0 |  | 1 | 86.52% | `lcasr/results/ctc_seq16384_unadapted_baseline/summary.csv` |
| 16384 no-adapt | earnings22 | test | 16384 | 14336 | 0 |  | 1 | 18.29% | `lcasr/results/ctc_seq16384_unadapted_baseline/summary.csv` |
| 16384 no-adapt | rev16 | test | 16384 | 14336 | 0 |  | 1 | 15.22% | `lcasr/results/ctc_seq16384_unadapted_baseline/summary.csv` |
| 16384 no-adapt | tedlium | test | 16384 | 14336 | 0 |  | 1 | 6.23% | `lcasr/results/ctc_seq16384_unadapted_baseline/summary.csv` |
| 65536 adapted stride2048 | chime6 | test | 65536 | 63488 | 5 | 9em5 | 1 | 75.54% | `lcasr/results/seq_65536_investigation/self_training_stride2048/summary_by_setting.csv` |
| 65536 adapted stride2048 | earnings22 | test | 65536 | 63488 | 5 | 9em5 | 1 | 14.83% | `lcasr/results/seq_65536_investigation/self_training_stride2048/summary_by_setting.csv` |
| 65536 adapted stride2048 | rev16 | test | 65536 | 63488 | 5 | 9em5 | 1 | 14.13% | `lcasr/results/seq_65536_investigation/self_training_stride2048/summary_by_setting.csv` |
| 65536 adapted stride2048 | tedlium | test | 65536 | 63488 | 5 | 9em5 | 1 | 5.75% | `lcasr/results/seq_65536_investigation/self_training_stride2048/summary_by_setting.csv` |
| 65536 no-adapt | chime6 | test | 65536 | 57344 | 0 |  | 1 | 86.76% | `lcasr/results/seq_65536_investigation/unadapted_baseline/summary.csv` |
| 65536 no-adapt | earnings22 | test | 65536 | 57344 | 0 |  | 1 | 18.32% | `lcasr/results/seq_65536_investigation/unadapted_baseline/summary.csv` |
| 65536 no-adapt | rev16 | test | 65536 | 57344 | 0 |  | 1 | 15.21% | `lcasr/results/seq_65536_investigation/unadapted_baseline/summary.csv` |
| 65536 no-adapt | tedlium | test | 65536 | 57344 | 0 |  | 1 | 6.17% | `lcasr/results/seq_65536_investigation/unadapted_baseline/summary.csv` |
