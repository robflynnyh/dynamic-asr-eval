# ROB-55 Majority-Vote Research Plan

Budget target: finish the main conclusion within about one week of GPU time,
including reruns. Do not expand all axes at once.

## Stage 1: Vote Calibration

Goal: determine whether sampled teacher votes retain enough chunks and improve
TEDLIUM dev.

Default grid:

- `teacher_vote_num_samples=8`
- `teacher_vote_temperature in {0.7, 1.0}`
- `teacher_vote_min_count=3`
- `teacher_vote_similarity in {1.0, 0.9}`
- `training_mode in {teacher_ce, teacher_kl}`
- `optim_lr in {1e-7, 3e-7}`
- `augmentation in {freq3_width24_time0, no_aug}`

Decision rule:

- If no setting beats no-adapt on TEDLIUM dev, inspect logs for retained-label
  rate. If retention is too low, try `teacher_vote_min_count=2` or
  `teacher_vote_similarity=0.9` before increasing augmentation.
- If a setting improves dev by at least 1 absolute WER point, rerun the best
  2-3 settings with `REPEATS=3`.

## Stage 2: Robustness Check

Run the best Stage 1 settings on TEDLIUM test and Earnings22 test with three
repeats. Keep the same no-adapt beam baseline and report deltas only against
matching decode settings.

Decision rule:

- A useful outcome needs consistent improvement on at least one dataset.
- A strong outcome is at least 5% relative WER improvement on one dataset with
  repeat variance smaller than the mean gain.

## Stage 3: Method Expansion

Only after Stage 1 has a viable vote threshold:

- Try `grpo`/`maxrl` using the majority-vote transcript as the reward reference.
- Try weaker augmentation (`freq2_width16_time0`) if teacher forcing appears to
  erase the benefit of stronger masks.
- Try near-match vote clustering (`teacher_vote_similarity=0.9`) if exact
  agreement is too sparse but sampled transcripts are visibly close.

## Stop Criteria

- Stop early if majority voting either skips almost all chunks or repeatedly
  worsens dev WER after threshold relaxation.
- Do not run a full test-set grid until a dev setting has a plausible retained
  label rate and a matching dev improvement.

## Stage 1b Decision

Stage 1b triggered the stop criterion for the current formulation. Exact
matching gave only a `0.41%` relative best repeat-mean gain on TEDLIUM dev and
accepted only `22-23` vote updates over three repeats. Relaxed `0.9` similarity
accepted hundreds of updates but consistently worsened WER by `5.85%` to
`7.99%` relative. Do not expand this exact vote-selection method to test sets or
GRPO/MAXRL without changing the teacher agreement primitive.

## Stage 2 Decision

Stage 2 tested medoid representative selection and a mean-probability/entropy
confidence filter. It did not justify test-set expansion. The best setting was
`teacher_ce`, LR `1e-7`, similarity `0.95`, medoid representative, with WER
`0.111722` vs baseline `0.112026` (`-0.27%` relative), which is smaller than
the observed repeat standard deviation. Similarity `0.9` still worsened WER by
`2.64%` to `7.08%` relative.

## Stage 3: Final Threshold Check

Run one final TEDLIUM-dev threshold check before closing the majority-vote path:

- `teacher_vote_min_count=2`
- `teacher_vote_similarity in {1.0, 0.95}`
- `teacher_vote_num_samples=8`
- `teacher_vote_temperature=0.7`
- `training_mode in {teacher_ce, teacher_kl}`
- `optim_lr in {1e-7, 3e-7}`
- `augmentation=freq3_width24_time0`
- three repeats

Decision rule:

- If exact or `0.95` voting with min count `2` gives a clear repeat-mean dev
  gain of at least `1%` relative, rerun only the best setting on TEDLIUM test.
- If it is neutral or worse, stop. The evidence would then cover strict voting,
  relaxed voting, medoid representative selection, confidence filtering, and the
  main min-agreement threshold axis without spending test-set or reward-method
  budget.

## Stage 3 Decision

Stage 3 completed the final threshold check and triggered the stop rule. The
best repeat-mean setting was `teacher_kl`, LR `1e-7`, exact similarity `1.0`,
min count `2`, with WER `0.111455` vs baseline `0.112026` (`-0.51%`
relative). Exact voting remained too sparse, accepting only `27-34` updates per
setting (`3.0-3.7%`). Near-exact `0.95` voting accepted `353-374` updates per
setting (`38.6-40.9%`) but was neutral or worse, including `+5%` relative
degradations in two settings.

Stop the current majority-vote branch. Do not spend more GPU budget on
TEDLIUM-test, Earnings22, GRPO, or MAXRL expansion for this formulation. A
future attempt should change the agreement primitive or teacher confidence
signal rather than tune these thresholds further.

## Stage 4: Segmented Utterance Diagnostic

Robert's 2026-05-09 follow-up supersedes the Stage 3 stop decision: keep
researching until May 15 unless explicitly asked to stop, and investigate what
kind of teacher samples help before handing the issue back.

The next bounded experiment is not another broad grid. Run the current best
Stage 3 setting on TEDLIUM dev at STM-utterance granularity and record, for each
utterance:

- baseline WER before adaptation;
- adapted WER after one teacher-selected update pass;
- whether the utterance improved, worsened, or stayed unchanged;
- selected teacher-label WER against the utterance reference when an update is
  accepted;
- vote count, vote support, candidate texts, skip stage, and skip reason.

Default setting:

- `teacher_kl`, LR `1e-7`, KL temperature `1.0`
- `freq3_width24_time0`
- vote `N=8`, temp `0.7`, min count `2`, exact similarity `1.0`
- representative `medoid`
- TEDLIUM dev segmented from STM files (`507` utterances, about `1.6` hours of
  audio)

Expected runtime after the one-utterance smoke is roughly `1-2` GPU hours,
depending on queue wait and long-utterance overhead. The result should decide
whether high-quality teacher samples ever produce local gains and which
teacher-WER bands deserve a follow-up filter/tuning run.

Queued wrapper:

```bash
screen -L -Logfile lcasr/results/enc_dec/enc_dec_majority_vote_utterance_diagnostic/logs/screen_rob55_segmented_diagnostic.log \
  -dmS rob55_segmented_diagnostic \
  bash -lc '/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob55_segmented_diagnostic_queued.sh'
```

## Stage 4 Decision

Stage 4 completed successfully and wrote
`lcasr/results/enc_dec/enc_dec_majority_vote_utterance_diagnostic/summary.json`.
It processed `507` segmented TEDLIUM-dev utterances and found identical corpus
WER before and after adaptation: `0.115121 -> 0.115121`. Exact min-count-2
voting accepted `88` teacher updates and skipped `454`; `447` skips were
below-threshold vote agreement. Among selected teacher labels, `24` had WER
`<= 0.10` against the gold utterance and `50` had WER `<= 0.25`, but no
accepted update changed the beam-search output under the Stage 3 recipe.

Interpretation: the Stage 3 recipe is too weak or too sparse for segmented
utterance adaptation. Do not conclude from this alone that good teacher samples
cannot help; first test stronger local updates and augmentation settings.

## Stage 5: Segmented Update Sensitivity

Run a bounded sensitivity sweep on the segmented diagnostic:

- `teacher_ce` vs `teacher_kl`;
- LR `1e-6` and `3e-6`;
- one vs three epochs;
- no augmentation, `freq3_width24_time0`, and `freq6_width34_time0`;
- vote samples `8` vs `16`, temperature `0.7` vs `1.0`;
- exact vote similarity `1.0`, min count `2`, medoid representative.

Decision rule:

- If at least one setting improves a meaningful number of low-teacher-WER
  accepted utterances without a larger worsened count, use that setting to
  design the next full-recording dev run.
- If stronger updates still produce no movement or mostly worsen clean-label
  cases, pivot away from majority-vote teacher labels toward a different
  teacher signal or objective.

Queued wrapper:

```bash
screen -L -Logfile lcasr/results/enc_dec/enc_dec_majority_vote_segmented_sensitivity/logs/screen_rob55_segmented_sensitivity.log \
  -dmS rob55_segmented_sensitivity \
  bash -lc '/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob55_segmented_sensitivity_queued.sh'
```

## Stage 5 Decision

Stage 5 completed successfully and wrote per-setting summaries under
`lcasr/results/enc_dec/enc_dec_majority_vote_segmented_sensitivity/`. The
best corpus result was only `teacher_ce`, LR `1e-6`, one epoch, no augmentation,
vote `N=8`, temp `0.7`, at `0.114955` WER versus `0.115121` baseline
(`-0.14%` relative). Stronger settings moved more utterances but mostly balanced
improvements with regressions; `N=16`, temp `0.7` accepted the most labels and
worsened corpus WER by `+0.43%` relative.

This argues against simply increasing update strength, augmentation, or sample
count. Low teacher-label WER was also not a reliable sufficient condition for
improvement: in the best Stage 5 setting, accepted labels with WER `<=0.25`
improved two utterances and worsened two utterances.

## Stage 6: Deterministic-Anchored Vote Gate

Hypothesis: sampled majority labels may be lower quality than the deterministic
beam teacher label. Instead of selecting a sampled medoid label, use the
stochastic vote set as a confidence gate and train on the deterministic beam
label when that deterministic label has enough exact support.

Run a bounded TEDLIUM-dev full-recording sweep:

- `teacher_vote_representative_strategy=deterministic`;
- `teacher_vote_include_deterministic=1`;
- vote samples `8` and `16`;
- vote temperatures `0.7` and `1.0`;
- similarities `1.0` and `0.95`, min count `2`;
- `teacher_ce` at LR `1e-7` and `3e-7`;
- `grpo` and `maxrl` at LR `1e-7`;
- `freq3_width24_time0`, one epoch, one repeat.

Decision rule:

- If the deterministic-anchored gate gives a plausible TEDLIUM-dev gain, rerun
  only the best one or two settings with repeats before considering test sets.
- If CE remains tiny and GRPO/MAXRL are neutral or worse, treat the
  majority-vote family as unlikely to reach the requested `5%` relative target
  without a different teacher-quality signal.
