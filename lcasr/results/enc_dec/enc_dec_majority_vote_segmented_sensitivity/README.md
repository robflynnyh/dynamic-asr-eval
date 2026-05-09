# ROB-55 Stage 5 Segmented Sensitivity

This stage follows the Stage 4 segmented diagnostic. Stage 4 found accepted
teacher labels with good WER against gold, but the current Stage 3 recipe
(`teacher_kl`, LR `1e-7`, one epoch, weak frequency masking) produced no
utterance-level decode changes.

The goal here is to test whether clean accepted labels can help after changing
only the student update recipe and a small vote sample-count/temperature axis.
This is still TEDLIUM dev and remains diagnostic, not a test-set claim.

Queued wrapper:

```bash
screen -L -Logfile lcasr/results/enc_dec/enc_dec_majority_vote_segmented_sensitivity/logs/screen_rob55_segmented_sensitivity.log \
  -dmS rob55_segmented_sensitivity \
  bash -lc '/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob55_segmented_sensitivity_queued.sh'
```

Settings:

| Name | Mode | LR | Epochs | Augmentation | Vote samples | Vote temp |
|---|---|---:|---:|---|---:|---:|
| `ce_lr1e6_e1_noaug_vote8_t0p7` | `teacher_ce` | `1e-6` | `1` | none | `8` | `0.7` |
| `ce_lr1e6_e3_noaug_vote8_t0p7` | `teacher_ce` | `1e-6` | `3` | none | `8` | `0.7` |
| `ce_lr3e6_e1_noaug_vote8_t0p7` | `teacher_ce` | `3e-6` | `1` | none | `8` | `0.7` |
| `ce_lr1e6_e3_freq3_vote8_t0p7` | `teacher_ce` | `1e-6` | `3` | `freq3_width24_time0` | `8` | `0.7` |
| `ce_lr1e6_e3_freq6_vote8_t0p7` | `teacher_ce` | `1e-6` | `3` | `freq6_width34_time0` | `8` | `0.7` |
| `kl_lr1e6_e3_noaug_vote8_t0p7` | `teacher_kl` | `1e-6` | `3` | none | `8` | `0.7` |
| `ce_lr1e6_e3_noaug_vote16_t0p7` | `teacher_ce` | `1e-6` | `3` | none | `16` | `0.7` |
| `ce_lr1e6_e3_noaug_vote16_t1p0` | `teacher_ce` | `1e-6` | `3` | none | `16` | `1.0` |

All settings use exact majority-vote similarity `1.0`, min count `2`, medoid
representative selection, beam width `5`, and length penalty `0.5`.

Inspect each setting's `summary.json`, `summary.csv`, `utterance_diagnostics.jsonl`,
and `teacher_events.jsonl` after the callback.

Outcome: see `OUTCOME.md`. The best setting was only `-0.14%` relative WER, and
stronger settings mostly traded improvements for regressions.
