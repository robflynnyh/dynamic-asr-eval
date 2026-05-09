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

## 2026-05-09

- Recent Linear comment from Robert asked to continue the ROB-55 research rather
  than hand over. This supersedes the previous completion note.
- Continued from PR #8 on branch
  `symphony/ROB-55-majority-vote-self-training` at
  `fb8cd134d6d1fdd6eb6c8f7d59b90fc875de040c`.
- Stage 2 hypothesis: relaxed voting may have degraded because the selected
  support-cluster representative was arbitrary. Added
  `--teacher_vote_representative_strategy medoid`, which chooses the vote
  candidate with the highest mean similarity to the largest support cluster.
- Added launcher support for recording representative strategy and optional
  low-confidence thresholds in the majority-vote run names/logs.
- Added `scripts/run_rob55_majority_vote_stage2_medoid_confidence_queued.sh`
  for a bounded TEDLIUM-dev run: `teacher_ce`/`teacher_kl`, LRs `1e-7`/`3e-7`,
  `freq3_width24_time0`, vote `N=8`, temp `0.7`, min count `3`, similarities
  `0.9`/`0.95`, medoid representative, low-confidence filter enabled, two
  repeats.
- Validation before queueing: copied the ignored local `paths.yaml` into this
  Symphony checkout from `/exp/exp4/acp21rjf/dynamic-asr-eval/paths.yaml`;
  py_compile and shell syntax checks passed; the Stage 2 wrapper callback dry
  run exercised the `EXIT` trap; a real one-recording TEDLIUM-dev smoke through
  `/store/store5/software/simple-gpu-schedule/with-gpu 1,2` loaded
  `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt`,
  selected medoid vote labels, applied the low-confidence filter, completed CE
  updates, decoded, and wrote
  `/exp/exp4/acp21rjf/.scratch/rob55_stage2_medoid_smoke_1.pkl` with WER
  `0.1326530612244898`.
- Stage 2 completed successfully via callback and wrote
  `lcasr/results/enc_dec/enc_dec_majority_vote_stage2_medoid_confidence/summary.csv`.
  Best repeat-mean WER was `0.111722` for `teacher_ce`, LR `1e-7`,
  similarity `0.95`, medoid representative, vs no-adapt baseline `0.112026`
  (`-0.27%` relative), smaller than repeat variance. Similarity `0.9`
  accepted many more updates but still worsened WER by `2.64%` to `7.08%`
  relative. Do not expand this setting to test sets.
- Added Stage 3 min-count-2 threshold check wrapper
  `scripts/run_rob55_majority_vote_stage3_min_count2_queued.sh`. This is the
  final bounded TEDLIUM-dev check for the majority-vote threshold axis before
  stopping or expanding only one clearly positive setting.
- Stage 3 completed successfully via callback and wrote
  `lcasr/results/enc_dec/enc_dec_majority_vote_stage3_min_count2/summary.csv`.
  Best repeat-mean WER was `0.111455` for `teacher_kl`, LR `1e-7`, exact
  similarity `1.0`, min count `2`, vs no-adapt baseline `0.112026`
  (`-0.51%` relative). Exact voting accepted only `27-34` updates per setting
  (`3.0-3.7%`). Near-exact `0.95` voting accepted `353-374` updates per setting
  (`38.6-40.9%`) but was neutral or worse. Final decision for ROB-55: stop this
  majority-vote formulation rather than expand to test sets, Earnings22, GRPO,
  or MAXRL; future work needs a different agreement primitive or teacher
  confidence signal.
- Robert's 2026-05-09 07:49 UTC Linear comment superseded the Stage 3 stop
  decision: do not hand the issue back before May 15 unless asked, continue
  exploring majority-vote axes and alternative diagnostics, and if the current
  method looks weak, identify which teacher samples help.
- Added optional teacher-update diagnostics to `lcasr/lib.py`; normal runs are
  unchanged unless `teacher_diagnostics_path` is set. Diagnostic events record
  accepted/skipped teacher updates, vote candidates/support, vote count,
  selected teacher text, confidence fields, CTC/decode agreement fields, and
  skip reasons.
- Added
  `lcasr/results/enc_dec/enc_dec_majority_vote/run_segmented_tedlium_diagnostic.py`
  and `scripts/run_rob55_segmented_diagnostic_queued.sh`. The runner steps
  through TEDLIUM dev STM utterances, compares baseline vs adapted WER per
  utterance, records selected teacher-label WER against gold, and writes
  `utterance_diagnostics.jsonl`, `teacher_events.jsonl`, `summary.csv`, and
  `summary.json`.
- Validation before queueing: `python3.10 -m py_compile` passed for
  `lcasr/lib.py`, `lcasr/enc_dec_dynamic_eval_test.py`, and the segmented
  diagnostic runner; `bash -n scripts/run_rob55_segmented_diagnostic_queued.sh`
  passed; callback dry-run through the wrapper `EXIT` trap succeeded; a real
  one-utterance smoke through `/store/store5/software/simple-gpu-schedule/with-gpu
  1,2` loaded the real checkpoint on GPU 1, processed
  `AlGore_2009:0000:13.04-23.46`, wrote scratch diagnostic artifacts, and
  exited with status `0`.
- Stage 4 completed successfully via callback. The segmented diagnostic
  processed `507` TEDLIUM-dev utterances and wrote
  `lcasr/results/enc_dec/enc_dec_majority_vote_utterance_diagnostic/summary.json`.
  Baseline and adapted corpus WER were identical at `0.115121`; no utterances
  improved or worsened. Exact min-count-2 voting accepted `88` teacher updates
  and skipped `454`; `24` selected labels had WER `<=0.10` against gold and
  `50` had WER `<=0.25`, but none moved the beam output under `teacher_kl`,
  LR `1e-7`, one epoch, and `freq3_width24_time0`.
- Added
  `lcasr/results/enc_dec/enc_dec_majority_vote_utterance_diagnostic/OUTCOME.md`,
  `lcasr/results/enc_dec/enc_dec_majority_vote_segmented_sensitivity/README.md`,
  and `scripts/run_rob55_segmented_sensitivity_queued.sh`. Stage 5 will keep
  the same segmented diagnostic while varying CE/KL, LR, epochs, augmentation,
  vote sample count, and vote temperature to test whether accepted clean
  teacher labels can help under stronger local updates.
- Stage 5 validation before queueing: `bash -n
  scripts/run_rob55_segmented_sensitivity_queued.sh` passed; `git diff --check`
  passed; callback dry-run through the wrapper `EXIT` trap succeeded; a real
  one-setting, one-utterance smoke through
  `/store/store5/software/simple-gpu-schedule/with-gpu 1,2` acquired GPU 1,
  loaded the real checkpoint, wrote scratch artifacts under
  `/exp/exp4/acp21rjf/.scratch/rob55_segmented_sensitivity_smoke`, and exited
  with status `0`.
- Stage 5 completed successfully via callback. The best segmented corpus result
  was `teacher_ce`, LR `1e-6`, one epoch, no augmentation, vote `N=8`, temp
  `0.7`: `0.114955` WER vs `0.115121` baseline (`-0.14%` relative). Stronger
  CE settings moved more utterances but produced as many or more worsened cases
  than improvements; `N=16`, temp `0.7` accepted `179` labels and worsened WER
  by `+0.43%` relative. Low teacher-label WER was not sufficient for gains: in
  the best setting, accepted labels with teacher WER `<=0.25` improved two
  utterances and worsened two.
- Added deterministic-anchored voting via
  `teacher_vote_representative_strategy=deterministic`. This uses stochastic
  samples as a confidence gate but trains on the deterministic beam teacher
  label when it has enough support. Added
  `scripts/run_rob55_majority_vote_stage6_deterministic_anchor_queued.sh` for a
  bounded TEDLIUM-dev sweep over vote samples `8/16`, temperatures `0.7/1.0`,
  similarities `1.0/0.95`, `teacher_ce`, `grpo`, and `maxrl`.
