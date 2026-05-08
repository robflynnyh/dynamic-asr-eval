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
