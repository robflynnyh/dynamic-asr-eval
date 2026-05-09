# ROB-55 Stage 4 Segmented TEDLIUM Diagnostic

Date: 2026-05-09

This diagnostic followed Robert's 2026-05-09 request to inspect which teacher
samples help before stopping the ROB-55 research. It ran the current best
Stage 3 majority-vote setting at STM-utterance granularity on TEDLIUM dev:

- `teacher_kl`, LR `1e-7`, KL temperature `1.0`
- one epoch
- augmentation `freq3_width24_time0`
- vote `N=8`, temperature `0.7`, min count `2`
- exact vote similarity `1.0`
- representative `medoid`

Outputs:

- `utterance_diagnostics.jsonl`
- `teacher_events.jsonl`
- `summary.csv`
- `summary.json`

The queued callback reported exit status `0`.

## Result

The segmented run processed `507` TEDLIUM-dev utterances. Corpus WER was
unchanged:

| Metric | Value |
|---|---:|
| Baseline corpus WER | `0.115121` |
| Adapted corpus WER | `0.115121` |
| Improved utterances | `0` |
| Worsened utterances | `0` |
| Unchanged utterances | `507` |

Teacher-update retention:

| Event | Count |
|---|---:|
| Accepted teacher updates | `88` |
| Skipped teacher updates | `454` |
| Below-threshold vote skips | `447` |

Accepted-vote support was usually weak:

| Vote count out of 8 | Accepted events |
|---:|---:|
| `2` | `45` |
| `3` | `13` |
| `4` | `15` |
| `5` | `6` |
| `6` | `3` |
| `7` | `3` |
| `8` | `3` |

Selected teacher-label WER against gold was often reasonable but not enough to
move the decoded output under this setting:

| Teacher-label WER band | Utterances |
|---|---:|
| exact `0.0` | `7` |
| `<= 0.10` | `24` |
| `<= 0.25` | `50` |
| `<= 0.50` | `65` |
| `> 0.50` | `16` |

## Interpretation

This result should not be read as proof that good teacher samples are useless.
It shows that the Stage 3 update strength is too weak or too sparse to change
segmented beam-search outputs, even when the accepted teacher label is close to
the gold transcript.

The next diagnostic should hold the same segmented setup and vary only the
student update recipe: higher LR, more epochs, hard CE vs KL, no augmentation
vs weak/stronger frequency masking, and a small vote-sample-count/temperature
check. If good teacher labels still fail to move utterance-level WER under
stronger local updates, the evidence will support pivoting away from this
majority-vote self-training path.
