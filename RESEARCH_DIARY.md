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
- Callback incident: ROB-51 completed successfully, but its Linear completion
  callback failed because the wrapper had `cd`ed into a subdirectory and then
  called `scripts/linear_experiment_callback.py` with a cwd-relative path. ROB-55
  had the same bug in its live wrapper. A compatibility shim was added at
  `lcasr/scripts/linear_experiment_callback.py` for the already-running ROB-55
  process, and `scripts/run_rob55_majority_vote_initial_sweep_queued.sh` was
  patched so future exits `cd "${REPO_ROOT}"` before calling the callback.
- Future agents: always smoke test detached experiment callbacks before queueing
  a run. At minimum, run the callback helper with `--dry-run` from the exact cwd
  the wrapper will have at exit, using the same `--log` and `--results` paths
  passed by the wrapper.
- 2026-05-08 ROB-55 Stage 1 analysis: the initial TEDLIUM-dev majority-vote
  sweep produced all 33 pickles despite the wrapper exit 127. Manual aggregation
  wrote `lcasr/results/enc_dec/enc_dec_majority_vote/summary.csv`. Baseline WER
  was `0.11203`; best setting was `teacher_ce`, LR `3e-7`,
  `freq3_width24_time0`, vote `N=8`, temp `0.7`, min count `3`, similarity
  `0.9`, WER `0.10970` (`-0.00232` absolute, `-2.07%` relative). This is a
  small signal, not a 5% relative gain. Added a Stage 1b queued wrapper to rerun
  the plausible frequency-mask/temp-0.7 axis with three repeats before any
  test-set expansion.
- 2026-05-08 ROB-55 Stage 1b analysis: the callback-backed repeat check
  completed successfully with exit status `0` and wrote
  `lcasr/results/enc_dec/enc_dec_majority_vote_stage1b/summary.csv`. The best
  repeat-mean setting was `teacher_kl`, LR `3e-7`, exact vote similarity `1.0`,
  WER `0.111566` (`-0.000461` absolute, `-0.41%` relative). Exact matching
  accepted only `22-23` vote updates across three repeats and skipped `892-893`
  as below threshold. Relaxing to similarity `0.9` accepted `558-582` updates
  but consistently worsened WER by `5.85%` to `7.99%` relative. Current
  conclusion: do not expand this exact majority-vote teacher formulation to test
  sets or GRPO/MAXRL; it needs a different agreement primitive or stronger
  confidence filter first.
