# ROB-63 RL self-training comparison

Snapshot generated from completed ROB-63 pickles.
RL deltas compare each RL `step_30000` self-training cell against the matching old-seed cell.
Normal/unadapted deltas compare each self-training cell against the same checkpoint's normal decoding WER.

## Summary

- Completed 16 one-epoch result rows covering 2 datasets, 2 splits, 2 checkpoints, 1 training modes, 2 learning rates, 2 frequency-mask settings.
- Teacher filtering is encoded in the augmentation label when present; `basic_repeat_filter` rows use the light repeat/length teacher filter. `Delta vs old seed` is only meaningful for the RL rows because old-seed rows are the matching-cell reference.
- chime6/dev: RL `step_30000` beats the matching old-seed cell in 2/4 cells; RL-vs-old absolute WER deltas span -0.00587 to +0.00458. Best RL cell is rl_step_30000 teacher_ce lr=1em8 freq9_width44_time0 at 0.99413 WER; best old-seed cell is old_seed teacher_ce lr=3em8 freq9_width44_time0 at 0.99411 WER. Relative change vs normal decoding spans +22.49% to +23.06% for RL and +19.14% to +19.85% for old seed.
- rev16/test: RL `step_30000` beats the matching old-seed cell in 4/4 cells; RL-vs-old absolute WER deltas span -0.06754 to -0.00270. Best RL cell is rl_step_30000 teacher_ce lr=3em8 freq9_width44_time0 at 0.17176 WER; best old-seed cell is old_seed teacher_ce lr=3em8 freq9_width44_time0 at 0.17446 WER. Relative change vs normal decoding spans -0.18% to +0.26% for RL and -1.23% to +35.91% for old seed.
- Sanity note: rev16 old_seed teacher_ce lr=1em8 freq9_width44_time0 is an outlier versus normal decoding (+35.91%, WER 0.24006); the error mix is dominated by deletion rate 0.14358.
- Sanity note: rev16 old_seed teacher_ce lr=1em8 freq9_width44_time0_basic_repeat_filter is an outlier versus normal decoding (+34.14%, WER 0.23693); the error mix is dominated by deletion rate 0.14502.

## Unadapted vs adapted WER

This is the main readout. `Old normal WER` and `RL normal WER` are the unadapted beam5/lp0.5 decoding baselines for each checkpoint. The adapted columns are the one-epoch self-training WERs for the listed setting.

| Dataset | Split | Mode | LR | Augmentation | Old normal WER | Old adapted WER | Old adapted vs normal | RL normal WER | RL adapted WER | RL adapted vs normal | RL adapted vs old adapted |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| chime6 | dev | teacher_ce | 1em8 | freq9_width44_time0 | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 0.99413 | +0.18255 | -0.00587 |
| chime6 | dev | teacher_ce | 1em8 | freq9_width44_time0_basic_repeat_filter | 0.83439 | 1.00000 | +0.16561 | 0.81157 | 0.99648 | +0.18491 | -0.00352 |
| chime6 | dev | teacher_ce | 3em8 | freq9_width44_time0 | 0.83439 | 0.99411 | +0.15972 | 0.81157 | 0.99869 | +0.18712 | +0.00458 |
| chime6 | dev | teacher_ce | 3em8 | freq9_width44_time0_basic_repeat_filter | 0.83439 | 0.99514 | +0.16075 | 0.81157 | 0.99702 | +0.18545 | +0.00188 |
| rev16 | test | teacher_ce | 1em8 | freq9_width44_time0 | 0.17663 | 0.24006 | +0.06343 | 0.17206 | 0.17252 | +0.00045 | -0.06754 |
| rev16 | test | teacher_ce | 1em8 | freq9_width44_time0_basic_repeat_filter | 0.17663 | 0.23693 | +0.06030 | 0.17206 | 0.17178 | -0.00029 | -0.06515 |
| rev16 | test | teacher_ce | 3em8 | freq9_width44_time0 | 0.17663 | 0.17446 | -0.00217 | 0.17206 | 0.17176 | -0.00030 | -0.00270 |
| rev16 | test | teacher_ce | 3em8 | freq9_width44_time0_basic_repeat_filter | 0.17663 | 0.17524 | -0.00139 | 0.17206 | 0.17204 | -0.00003 | -0.00320 |

## Full table

| Dataset | Split | Mode | LR | Augmentation | Checkpoint | Adapted WER | Unadapted WER | Delta vs unadapted | Relative vs unadapted | Delta vs old seed | Relative vs old seed | Ins | Del | Sub |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chime6 | dev | teacher_ce | 1em8 | freq9_width44_time0 | old_seed | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | freq9_width44_time0 | rl_step_30000 | 0.99413 | 0.81157 | +0.18255 | +22.49% | -0.00587 | -0.59% | 0.00002 | 0.99329 | 0.00082 |
| chime6 | dev | teacher_ce | 1em8 | freq9_width44_time0_basic_repeat_filter | old_seed | 1.00000 | 0.83439 | +0.16561 | +19.85% | +0.00000 | +0.00% | 0.00000 | 1.00000 | 0.00000 |
| chime6 | dev | teacher_ce | 1em8 | freq9_width44_time0_basic_repeat_filter | rl_step_30000 | 0.99648 | 0.81157 | +0.18491 | +22.78% | -0.00352 | -0.35% | 0.00003 | 0.99617 | 0.00028 |
| chime6 | dev | teacher_ce | 3em8 | freq9_width44_time0 | old_seed | 0.99411 | 0.83439 | +0.15972 | +19.14% | +0.00000 | +0.00% | 0.00000 | 0.99339 | 0.00072 |
| chime6 | dev | teacher_ce | 3em8 | freq9_width44_time0 | rl_step_30000 | 0.99869 | 0.81157 | +0.18712 | +23.06% | +0.00458 | +0.46% | 0.00000 | 0.99863 | 0.00007 |
| chime6 | dev | teacher_ce | 3em8 | freq9_width44_time0_basic_repeat_filter | old_seed | 0.99514 | 0.83439 | +0.16075 | +19.27% | +0.00000 | +0.00% | 0.00002 | 0.99444 | 0.00069 |
| chime6 | dev | teacher_ce | 3em8 | freq9_width44_time0_basic_repeat_filter | rl_step_30000 | 0.99702 | 0.81157 | +0.18545 | +22.85% | +0.00188 | +0.19% | 0.00000 | 0.99643 | 0.00059 |
| rev16 | test | teacher_ce | 1em8 | freq9_width44_time0 | old_seed | 0.24006 | 0.17663 | +0.06343 | +35.91% | +0.00000 | +0.00% | 0.02233 | 0.14358 | 0.07415 |
| rev16 | test | teacher_ce | 1em8 | freq9_width44_time0 | rl_step_30000 | 0.17252 | 0.17206 | +0.00045 | +0.26% | -0.06754 | -28.14% | 0.02982 | 0.06022 | 0.08248 |
| rev16 | test | teacher_ce | 1em8 | freq9_width44_time0_basic_repeat_filter | old_seed | 0.23693 | 0.17663 | +0.06030 | +34.14% | +0.00000 | +0.00% | 0.01925 | 0.14502 | 0.07265 |
| rev16 | test | teacher_ce | 1em8 | freq9_width44_time0_basic_repeat_filter | rl_step_30000 | 0.17178 | 0.17206 | -0.00029 | -0.17% | -0.06515 | -27.50% | 0.03006 | 0.06017 | 0.08155 |
| rev16 | test | teacher_ce | 3em8 | freq9_width44_time0 | old_seed | 0.17446 | 0.17663 | -0.00217 | -1.23% | +0.00000 | +0.00% | 0.02987 | 0.06226 | 0.08234 |
| rev16 | test | teacher_ce | 3em8 | freq9_width44_time0 | rl_step_30000 | 0.17176 | 0.17206 | -0.00030 | -0.18% | -0.00270 | -1.55% | 0.02974 | 0.05906 | 0.08296 |
| rev16 | test | teacher_ce | 3em8 | freq9_width44_time0_basic_repeat_filter | old_seed | 0.17524 | 0.17663 | -0.00139 | -0.79% | +0.00000 | +0.00% | 0.02976 | 0.06302 | 0.08245 |
| rev16 | test | teacher_ce | 3em8 | freq9_width44_time0_basic_repeat_filter | rl_step_30000 | 0.17204 | 0.17206 | -0.00003 | -0.01% | -0.00320 | -1.82% | 0.02905 | 0.06120 | 0.08179 |
