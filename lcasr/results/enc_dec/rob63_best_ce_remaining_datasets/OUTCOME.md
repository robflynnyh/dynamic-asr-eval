# ROB-63 RL self-training comparison

Snapshot generated from completed ROB-63 pickles.
RL deltas compare each RL `step_30000` self-training cell against the matching old-seed cell.
Normal deltas compare each self-training cell against the same checkpoint's normal decoding WER.

## Summary

- Completed 4 one-epoch cells: 2 datasets x 2 checkpoints x 1 training modes x 1 learning rates x 1 frequency-mask settings.
- The sweep used no teacher filtering. `Delta vs old seed` is only meaningful for the RL rows because old-seed rows are the matching-cell reference.
- chime6: RL `step_30000` beats the matching old-seed cell in 0/1 cells; RL-vs-old absolute WER deltas span +0.00000 to +0.00000. Best RL cell is rl_step_30000 teacher_ce lr=1em7 freq3_width24_time0 at 1.00000 WER; best old-seed cell is old_seed teacher_ce lr=1em7 freq3_width24_time0 at 1.00000 WER. Matching normal-decoding baselines were not available for this dataset in the ROB-61 CSV.
- rev16: RL `step_30000` beats the matching old-seed cell in 1/1 cells; RL-vs-old absolute WER deltas span -0.00191 to -0.00191. Best RL cell is rl_step_30000 teacher_ce lr=1em7 freq3_width24_time0 at 0.23599 WER; best old-seed cell is old_seed teacher_ce lr=1em7 freq3_width24_time0 at 0.23790 WER. Matching normal-decoding baselines were not available for this dataset in the ROB-61 CSV.

## Full table

| Dataset | Mode | LR | Augmentation | Checkpoint | WER | Normal WER | Delta vs normal | Relative vs normal | Delta vs old seed | Relative vs old seed | Ins | Del | Sub |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chime6 | teacher_ce | 1em7 | freq3_width24_time0 | old_seed | 1.00000 |  |  |  | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | teacher_ce | 1em7 | freq3_width24_time0 | rl_step_30000 | 1.00000 |  |  |  | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| rev16 | teacher_ce | 1em7 | freq3_width24_time0 | old_seed | 0.23790 |  |  |  | +0.00000 | +0.00% | 0.02323 | 0.14223 | 0.07243 |
| rev16 | teacher_ce | 1em7 | freq3_width24_time0 | rl_step_30000 | 0.23599 |  |  |  | -0.00191 | -0.80% | 0.02280 | 0.13812 | 0.07507 |
