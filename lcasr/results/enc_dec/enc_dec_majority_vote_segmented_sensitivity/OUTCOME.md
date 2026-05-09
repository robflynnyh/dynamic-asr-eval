# ROB-55 Stage 5 Segmented Sensitivity Outcome

Date: 2026-05-09

Stage 5 tested whether the clean accepted labels found in Stage 4 become useful
after changing the local student update recipe. It varied CE/KL, learning rate,
epochs, augmentation strength, vote sample count, and vote temperature on the
segmented TEDLIUM-dev diagnostic.

Baseline segmented corpus WER was `0.115121`.

| Setting | WER | Relative delta | Improved / worsened | Accepted updates |
|---|---:|---:|---:|---:|
| `ce_lr1e6_e1_noaug_vote8_t0p7` | `0.114955` | `-0.14%` | `5 / 5` | `82` |
| `ce_lr1e6_e3_freq6_vote8_t0p7` | `0.115121` | `+0.00%` | `4 / 5` | `116` |
| `ce_lr1e6_e3_noaug_vote16_t1p0` | `0.115121` | `+0.00%` | `2 / 2` | `25` |
| `kl_lr1e6_e3_noaug_vote8_t0p7` | `0.115121` | `+0.00%` | `0 / 0` | `126` |
| `ce_lr1e6_e3_noaug_vote8_t0p7` | `0.115342` | `+0.19%` | `9 / 10` | `128` |
| `ce_lr3e6_e1_noaug_vote8_t0p7` | `0.115342` | `+0.19%` | `4 / 8` | `81` |
| `ce_lr1e6_e3_freq3_vote8_t0p7` | `0.115453` | `+0.29%` | `6 / 10` | `130` |
| `ce_lr1e6_e3_noaug_vote16_t0p7` | `0.115618` | `+0.43%` | `16 / 22` | `179` |

Interpretation:

- Stronger CE updates can move segmented utterance outputs, but the direction is
  not reliably positive.
- Extra epochs, stronger frequency masking, and more sampled votes did not help
  the corpus WER.
- Low teacher-label WER was not sufficient for improvement. In the best setting,
  labels with teacher WER `<=0.25` improved two utterances and worsened two.
- KL remained too conservative in this segmented setup: it accepted labels but
  produced no output changes.

Next step: pivot from selecting sampled majority labels to using samples as a
confidence gate for deterministic beam labels. Stage 6 runs this
deterministic-anchored vote variant on full TEDLIUM-dev recordings with CE,
GRPO, and MAXRL.
