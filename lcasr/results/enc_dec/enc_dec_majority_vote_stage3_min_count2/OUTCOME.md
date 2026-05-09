# ROB-55 Majority-Vote Stage 3 Min-Count-2 Sweep

Date: 2026-05-09

This sweep tested the final bounded threshold question for the current
majority-vote teacher formulation: whether dropping the vote minimum from `3`
to `2` could retain enough clean sampled teacher updates without broadening to
the harmful `0.9` near-match threshold.

The run used TEDLIUM dev with three repeats:

- `training_mode in {teacher_ce, teacher_kl}`
- LR in `{1e-7, 3e-7}`
- augmentation `freq3_width24_time0`
- vote `N=8`, temp `0.7`, min count `2`
- vote similarity in `{1.0, 0.95}`
- vote representative `medoid`

The queued callback reported exit status `0`. Aggregation output is in
`summary.csv`.

## Result

Baseline:

- `tedlium-dev-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug`
- WER: `0.112026`

Best repeat-mean setting:

- `teacher_kl`, LR `1e-7`, `freq3_width24_time0`, vote `N=8`, temp `0.7`,
  min count `2`, exact similarity `1.0`, medoid representative
- WER: `0.111455`
- WER std: `0.000158`
- Delta vs baseline: `-0.000571` absolute, `-0.51%` relative

Exact `1.0` similarity was safe but only marginally positive:

| Mode | LR | Similarity | WER | Delta vs baseline | Relative delta |
|---|---:|---:|---:|---:|---:|
| `teacher_kl` | `1e-7` | `1.0` | `0.111455` | `-0.000571` | `-0.51%` |
| `teacher_kl` | `3e-7` | `1.0` | `0.111602` | `-0.000424` | `-0.38%` |
| `teacher_ce` | `1e-7` | `1.0` | `0.111713` | `-0.000313` | `-0.28%` |
| `teacher_ce` | `3e-7` | `1.0` | `0.111971` | `-0.000055` | `-0.05%` |

Near-exact `0.95` similarity was mostly neutral or worse:

| Mode | LR | Similarity | WER | Delta vs baseline | Relative delta |
|---|---:|---:|---:|---:|---:|
| `teacher_ce` | `3e-7` | `0.95` | `0.112118` | `+0.000092` | `+0.08%` |
| `teacher_kl` | `1e-7` | `0.95` | `0.113924` | `+0.001897` | `+1.69%` |
| `teacher_ce` | `1e-7` | `0.95` | `0.117645` | `+0.005619` | `+5.02%` |
| `teacher_kl` | `3e-7` | `0.95` | `0.118198` | `+0.006171` | `+5.51%` |

## Vote Diagnostics

Vote-retention counts from the logs:

| Similarity family | Accepted vote updates | Below-threshold skips | Accept rate |
|---|---:|---:|---:|
| `1.0` | `27-34` per setting | `881-888` per setting | `3.0-3.7%` |
| `0.95` | `353-374` per setting | `541-562` per setting | `38.6-40.9%` |

Aggregated across settings, exact `1.0` voting accepted `120` of `3660`
attempted updates (`3.3%`). Near-exact `0.95` voting accepted `1455` of `3660`
attempted updates (`39.8%`), but the extra pseudo-labels did not translate to
reliable WER improvement.

## Interpretation

Lowering the minimum agreement threshold to `2` does not rescue the current
majority-vote teacher formulation. Exact voting remains too sparse for
meaningful adaptation, and relaxing similarity enough to retain more updates
again introduces harmful pseudo-labels. The best repeat-mean gain is only
`0.51%` relative on TEDLIUM dev, far below the issue target of at least `5%`
relative on one dataset and below the pre-declared `1%` relative expansion
threshold.

No further GPU sweeps are recommended for this formulation. Future work should
change the agreement primitive or teacher signal rather than expanding this
branch to test sets, Earnings22, GRPO, or MAXRL.
