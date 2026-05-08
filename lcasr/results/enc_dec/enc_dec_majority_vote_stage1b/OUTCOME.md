# ROB-55 Majority-Vote Stage 1b Repeat Check

Date: 2026-05-08

This repeat check reran the plausible Stage 1 axis on TEDLIUM dev with three
repeats:

- `training_mode in {teacher_ce, teacher_kl}`
- LR in `{1e-7, 3e-7}`
- augmentation `freq3_width24_time0`
- vote `N=8`, temp `0.7`, min count `3`
- vote similarity in `{1.0, 0.9}`

The queued callback reported exit status `0`. Aggregation output is in
`summary.csv`.

## Result

Baseline:

- `tedlium-dev-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug`
- WER: `0.112026`

Best repeat-mean setting:

- `teacher_kl`, LR `3e-7`, `freq3_width24_time0`, vote `N=8`, temp `0.7`,
  min count `3`, exact similarity `1.0`, `tau=1.0`
- WER: `0.111566`
- WER std: `0.000389`
- Delta vs baseline: `-0.000461` absolute, `-0.41%` relative

Other exact-vote settings were also only marginally positive:

| Mode | LR | Similarity | WER | Delta vs baseline | Relative delta |
|---|---:|---:|---:|---:|---:|
| `teacher_kl` | `3e-7` | `1.0` | `0.111566` | `-0.000461` | `-0.41%` |
| `teacher_kl` | `1e-7` | `1.0` | `0.111787` | `-0.000239` | `-0.21%` |
| `teacher_ce` | `3e-7` | `1.0` | `0.111805` | `-0.000221` | `-0.20%` |
| `teacher_ce` | `1e-7` | `1.0` | `0.111916` | `-0.000111` | `-0.10%` |

Relaxed near-match voting was consistently harmful:

| Mode | LR | Similarity | WER | Delta vs baseline | Relative delta |
|---|---:|---:|---:|---:|---:|
| `teacher_ce` | `3e-7` | `0.9` | `0.118584` | `+0.006558` | `+5.85%` |
| `teacher_kl` | `1e-7` | `0.9` | `0.119008` | `+0.006982` | `+6.23%` |
| `teacher_kl` | `3e-7` | `0.9` | `0.120390` | `+0.008364` | `+7.47%` |
| `teacher_ce` | `1e-7` | `0.9` | `0.120979` | `+0.008953` | `+7.99%` |

## Vote Diagnostics

Log counts across the three repeats show the tradeoff:

| Setting family | Accepted vote updates | Below-threshold skips | Interpretation |
|---|---:|---:|---|
| exact similarity `1.0` | `22-23` | `892-893` | Very conservative; mostly skips adaptation. |
| relaxed similarity `0.9` | `558-582` | `333-357` | Much higher update rate, but pseudo-label quality is poor enough to degrade WER. |

## Interpretation

The current majority-vote path does not produce the requested kind of gain on
TEDLIUM dev. Exact matching is safe but too sparse to move WER meaningfully;
near-match clustering increases the amount of training signal but appears to add
bad pseudo-labels and consistently worsens the model.

Given the Stage 1 and Stage 1b results, expanding this exact formulation to
test-set runs or GRPO/MAXRL is not justified. A future attempt should change
the teacher selection primitive rather than only adding more GPU budget: for
example, normalize agreement at the token/posterior level, use edit-distance
clustering with a representative closest to the cluster center, or combine vote
agreement with a stronger confidence filter.
