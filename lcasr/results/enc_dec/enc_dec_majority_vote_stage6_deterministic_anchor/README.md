# ROB-55 Stage 6 Deterministic-Anchored Vote Gate

This stage tests an alternative to selecting sampled majority-vote labels. The
sampled decodes are used as a confidence gate, but the training target is the
deterministic beam teacher label when that deterministic label has enough exact
support among the samples.

Queued wrapper:

```bash
screen -L -Logfile lcasr/results/enc_dec/enc_dec_majority_vote_stage6_deterministic_anchor/logs/screen_rob55_majority_vote_stage6_deterministic_anchor.log \
  -dmS rob55_majority_vote_stage6_deterministic_anchor \
  bash -lc '/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob55_majority_vote_stage6_deterministic_anchor_queued.sh'
```

Settings:

- `teacher_vote_representative_strategy=deterministic`
- `teacher_vote_include_deterministic=1`
- vote samples `8` and `16`
- vote temperatures `0.7` and `1.0`
- vote similarities `1.0` and `0.95`, min count `2`
- `teacher_ce` at LR `1e-7` and `3e-7`
- `grpo` and `maxrl` at LR `1e-7`
- `freq3_width24_time0`, one epoch, one repeat

Inspect `summary.csv` after the callback.
