# ROB-55 Stage 7 CE Exact-Gate Repeat Sweep

Status: cancelled before start.

Robert's 2026-05-10 Linear comment asked to end the investigation and minimize
unnecessary changes in the final commit. The queued Stage 7 screen was stopped
while still waiting inside `with-gpu`; the wrapper did not start, no GPU work
ran, and no Stage 7 result log or summary was produced.

Stage 7 follows the Stage 6 deterministic-anchor result. It tests whether the
best Stage 6 signal is stable across repeats and whether CE update strength or
augmentation explains the gain.

Queued wrapper:

```bash
screen -L -Logfile lcasr/results/enc_dec/enc_dec_majority_vote_stage7_ce_exact_repeats/logs/screen_rob55_majority_vote_stage7_ce_exact_repeats.log \
  -dmS rob55_majority_vote_stage7_ce_exact_repeats \
  bash -lc '/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob55_majority_vote_stage7_ce_exact_repeats_queued.sh'
```

Settings:

- `teacher_vote_representative_strategy=deterministic`
- `teacher_vote_include_deterministic=1`
- vote similarities `1.0` only, min count `2`
- vote samples `8` and `16`
- vote temperature `0.7`
- `teacher_ce` only
- LRs `1e-7`, `3e-7`, and `1e-6`
- augmentations `freq3_width24_time0` and `no_aug`
- three repeats per setting

Rationale:

- Stage 6 best result was `teacher_ce`, LR `3e-7`, `N=16`, temp `0.7`,
  exact support, WER `0.110921` vs `0.112026` baseline (`-0.99%` relative).
- The relaxed `0.95` gate was harmful, so this sweep keeps exact agreement.
- GRPO and MAXRL were close to neutral in Stage 6, so this sweep spends the
  budget on CE repeat stability and update-strength sensitivity.

After callback completion, aggregate with:

```bash
python lcasr/results/enc_dec/enc_dec_majority_vote/aggregate.py \
  --directory lcasr/results/enc_dec/enc_dec_majority_vote_stage7_ce_exact_repeats \
  --csv lcasr/results/enc_dec/enc_dec_majority_vote_stage7_ce_exact_repeats/summary.csv
```
