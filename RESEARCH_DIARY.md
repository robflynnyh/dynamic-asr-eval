# Research Diary

## 2026-05-08

- ROB-55 started from `main` at `6b3b5309fddc03c3af7a294c34898a9f1ccd1b4a`
  on branch `symphony/ROB-55-majority-vote-self-training`.
- Recent Linear comments were empty, so the issue description is the active
  constraint.
- Added the initial plan to Linear: implement sampled teacher majority voting,
  document a staged research plan, validate the callback path, then queue a
  bounded first sweep rather than waiting in-agent.
- Validated the new majority-vote path with a one-recording TEDLIUM-dev smoke
  on GPU 2 using `teacher_vote_num_samples=2`, `teacher_vote_min_count=1`, and
  `training_mode=teacher_ce`. The run loaded the real encoder-decoder
  checkpoint, selected sampled teacher labels, performed CE updates, decoded
  the adapted recording, and wrote
  `/exp/exp4/acp21rjf/.scratch/rob55_majority_vote_smoke_1.pkl` with WER
  `0.14540816326530612`.
