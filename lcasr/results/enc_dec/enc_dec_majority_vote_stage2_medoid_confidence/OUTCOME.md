# ROB-55 Majority-Vote Stage 2 Medoid/Confidence Sweep

Date: 2026-05-09

This sweep tested whether relaxed sampled-teacher voting failed because the
representative transcript was arbitrary. The run used TEDLIUM dev with two
repeats:

- `training_mode in {teacher_ce, teacher_kl}`
- LR in `{1e-7, 3e-7}`
- augmentation `freq3_width24_time0`
- vote `N=8`, temp `0.7`, min count `3`
- vote similarity in `{0.9, 0.95}`
- vote representative `medoid`
- low-confidence filter enabled with mean max prob `>=0.35` and mean entropy
  `<=2.5`

The queued callback reported exit status `0`. Aggregation output is in
`summary.csv`.

## Result

Baseline:

- `tedlium-dev-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug`
- WER: `0.112026`

Best repeat-mean setting:

- `teacher_ce`, LR `1e-7`, `freq3_width24_time0`, vote `N=8`, temp `0.7`,
  min count `3`, similarity `0.95`, medoid representative
- WER: `0.111722`
- WER std: `0.000912`
- Delta vs baseline: `-0.000304` absolute, `-0.27%` relative

`0.95` similarity with medoid selection was mostly neutral to worse:

| Mode | LR | Similarity | WER | Delta vs baseline | Relative delta |
|---|---:|---:|---:|---:|---:|
| `teacher_ce` | `1e-7` | `0.95` | `0.111722` | `-0.000304` | `-0.27%` |
| `teacher_kl` | `1e-7` | `0.95` | `0.112800` | `+0.000774` | `+0.69%` |
| `teacher_ce` | `3e-7` | `0.95` | `0.113242` | `+0.001216` | `+1.09%` |
| `teacher_kl` | `3e-7` | `0.95` | `0.115066` | `+0.003040` | `+2.71%` |

`0.9` similarity still degraded WER despite medoid representative selection:

| Mode | LR | Similarity | WER | Delta vs baseline | Relative delta |
|---|---:|---:|---:|---:|---:|
| `teacher_kl` | `1e-7` | `0.9` | `0.114983` | `+0.002957` | `+2.64%` |
| `teacher_ce` | `3e-7` | `0.9` | `0.117829` | `+0.005803` | `+5.18%` |
| `teacher_kl` | `3e-7` | `0.9` | `0.118327` | `+0.006300` | `+5.62%` |
| `teacher_ce` | `1e-7` | `0.9` | `0.119957` | `+0.007931` | `+7.08%` |

## Vote Diagnostics

Vote-retention counts from the logs:

| Similarity family | Accepted vote updates | Below-threshold skips | Accept rate |
|---|---:|---:|---:|
| `0.95` | `186-192` | `418-424` | `30.5-31.5%` |
| `0.9` | `375-397` | `213-235` | `61.5-65.1%` |

The low-confidence filter did not appear as a dominant skip reason. The main
gate remained the vote threshold. Accepted `0.9` clusters had high support
counts, but the selected pseudo-labels were still harmful enough to increase
insertion errors and WER.

## Interpretation

Medoid representative selection plus the current confidence filter does not
rescue relaxed majority voting. The only positive setting is smaller than its
repeat standard deviation and far below the target `5%` relative gain. This is
not strong enough to justify a TEDLIUM-test or Earnings22 expansion.

The remaining bounded threshold question is whether exact or near-exact voting
with min count `2` can recover more clean updates than exact min count `3`
without the noise introduced by broad `0.9` matching.
