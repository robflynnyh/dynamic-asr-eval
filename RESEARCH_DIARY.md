# Research Diary

## 2026-05-21

- ROB-112: Added the 8192-context no-adapt CTC baseline scaffold for repeat 1 over Earnings-22, TEDLIUM, CHiME-6, and Rev16 test splits. The setup mirrors ROB-66/ROB-67 with `seq_len=8192`, `overlap=7168` (stride `1024`), checkpoint `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_8192_rp_1/step_105360.pt`, a callback-backed queued wrapper, and regenerated-summary/comparison helpers under `lcasr/results/ctc_seq8192_unadapted_baseline/`.
- ROB-67: Consolidated the completed 65536-context CTC result artifacts under one investigation root, `lcasr/results/seq_65536_investigation/`, after review feedback that the PR should not add several top-level result folders. The existing unadapted, `lr=1e-5`, higher-LR, longer-epoch, and stride-2048 artifacts were moved into subdirectories and the launch/finalizer defaults were updated to write to the new layout.
- ROB-67: Synced and aggregated the completed stride-2048 65536-context CTC self-training follow-up from Stanage. The final result set has all four expected PKLs under `lcasr/results/seq_65536_investigation/self_training_stride2048/`; the CHiME-6-only recovery job `10239452_0` completed with `MaxRSS=71739268K` under the higher `80G` CPU-memory request, and finalizer `10239453` verified `adapted_count=4/4`. Regenerated `summary.csv`, `summary_by_setting.csv`, and `summary.md` locally from the synced artifacts.

## 2026-05-20

- ROB-67: Diagnosed stride-2048 Stanage array `10235542` after its callback returned the issue to `Todo`. Earnings22, TEDLIUM, and Rev16 completed, but CHiME-6 task `10235542_2` hit Slurm `OUT_OF_MEMORY` at the 60G request with `MaxRSS=62914604K`; added a CHiME-6-only recovery launcher that preserves the existing three PKLs, requests higher CPU memory, and runs the finalizer expecting all four stride-2048 adapted outputs.

## 2026-05-18

- ROB-67: Inspected the completed stride-2048 CPU smoke `10227206`, which succeeded with WER `0.05072220356063151` but failed its Linear callback because the target state was malformed as `Todo scripts/run_rob67_ctc_seq65536_cpu_smoke.sbatch`. Patched the stride-2048 Stanage finalizer path to avoid exporting long callback strings through `sbatch --export`, added a finalizer callback-only smoke mode, and validated the stride-specific callback text locally before queueing the GPU array.

## 2026-05-16

- ROB-67: Added a callback-backed Stanage follow-up launcher for the requested final 65536-context CTC adapted eval using the 16384-style stride. The run writes to `lcasr/results/seq_65536_investigation/self_training_stride2048/` with `seq_len=65536`, `overlap=63488` (stride `2048`), adaptation epochs `5`, `lr=9e-5`, four test datasets, and a dependent Linear callback finalizer.
- ROB-67: Synced and aggregated the completed longer-epoch 65536-context CTC self-training follow-up from Stanage array `10220452` / finalizer `10220453`. The run produced all 16 expected adapted PKLs under `lcasr/results/seq_65536_investigation/self_training_longer_epochs/` for datasets `earnings22`, `tedlium`, `chime6`, and `rev16`, adaptation epochs `10` and `20`, and LRs `9e-5` and `3e-4`. Longer adaptation gives small best-row gains on TEDLIUM, Earnings22, and Rev16; CHiME-6 remains best with the earlier 5-epoch `lr=1e-5` row, and the CHiME-6 20-epoch `lr=3e-4` cell collapsed to 99.99% WER from near-total deletions.

## 2026-05-15

- ROB-67: Added a callback-backed Stanage follow-up launcher for the requested 10- and 20-epoch, `lr in {9e-5,3e-4}` 65536-context CTC adapted evaluation. The run writes to `lcasr/results/seq_65536_investigation/self_training_longer_epochs/` as a separate result family from the completed 1/5-epoch higher-LR sweep and reuses the validated ROB-67 cell runner plus finalizer callback path.
- ROB-67: Synced the completed Stanage higher-LR follow-up for the 65536-context CTC self-training eval from array `10169175` / finalizer `10169176`. The run produced all 16 expected PKLs under `lcasr/results/seq_65536_investigation/self_training_higher_lr/` for datasets `earnings22`, `tedlium`, `chime6`, and `rev16`, adaptation epochs `1` and `5`, and LRs `9e-5` and `3e-4`. Regenerated `summary.csv`, `summary_by_setting.csv`, and `summary.md` from the synced PKLs. Relative to the earlier `lr=1e-5` adapted run, the higher-LR epoch-5 rows improve TEDLIUM, Earnings22, and Rev16, while CHiME-6 remains best with `lr=1e-5`.

## 2026-05-12

- ROB-67: Inspected the completed Stanage finalizer output for the 65536-context CTC eval. The final run used `lr=1e-5` for adapted `epochs=1` and `epochs=5`, produced 4 unadapted PKLs and 8 adapted PKLs, and regenerated summaries under `lcasr/results/seq_65536_investigation/unadapted_baseline/` and `lcasr/results/seq_65536_investigation/self_training_lr1e5/`. Added a separate callback-backed Stanage follow-up sweep for higher LRs `9e-5` and `3e-4`, writing to `lcasr/results/seq_65536_investigation/self_training_higher_lr/` so those rows do not overwrite the completed `lr=1e-5` result set.

## 2026-05-11

- ROB-67: Prepared callback-backed Stanage scaffolding for the 65536-context CTC checkpoint. The setup uses `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_65536_rp_1/step_105360.pt` on Mimas and `/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_65536_rp_1/step_105360.pt` on Stanage, with `seq_len=65536`, `overlap=57344`, no-adapt plus self-training epochs `1` and `5`, and `lr=1e-5`. Added a CPU-only Stanage smoke job and a 12-cell Stanage GPU array with an `afterany` finalizer callback.

## 2026-05-20

- ROB-97: Added a callback-backed queue scaffold for the missing `enc_dec_v2`
  no-adapt beam5/lp0.5 CHiME-6 and Rev16 test-set rows required by ROB-96.
  The result path is `lcasr/results/enc_dec/enc_dec_v2/rob97_unadapted_beam/`
  and the queued wrapper notifies ROB-97 on every exit path, then ROB-96 on
  success.
- ROB-97: The queued `enc_dec_v2` no-adapt beam5/lp0.5 evals completed for
  `chime6/test` and `rev16/test` using
  `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt`.
  Regenerated `summary.csv` and `OUTCOME.md`; final WERs are `0.86331` for
  CHiME-6 test and `0.19109` for Rev16 test. The callback posted completion
  evidence to ROB-97, notified ROB-96, and set ROB-96 back to Todo.
- ROB-103: Added a time-mask-only scaled RMM eval scaffold under
  `lcasr/results/rmm_eval/ctc_seq16384_time_only_scaled_time_masks/`. The run
  uses the ROB-68 scaled time-mask count (`96` masks at `seq_len=16384`) but
  forces `rmm_branch='time'`, avoiding the frequency-only and time+frequency
  RMM branches.
- ROB-103: The first queued launch failed before evaluation because this fresh
  workspace's ignored `paths.yaml` lacked `datasets.rev16.test`. Made
  `lcasr/rev16/run.py` tolerate missing Rev16 config at import time, added the
  Rev16 key to `paths_template.yaml`, and populated the local ignored
  `paths.yaml` with the Mimas dataset paths for the rerun.

## 2026-05-19

- ROB-68: After a human follow-up requested dev results for the latest
  scaled-time-mask setup, prepared the existing
  `lcasr/results/rmm_eval/ctc_seq16384_scaled_time_masks/` package for mixed split summaries
  and queued a dev split pass for `earnings22`, `tedlium`, and `chime6`.
  `rev16` is excluded because the CTC dynamic-eval runner asserts that Rev16
  only supports `test`.
- ROB-68: Fixed the scaled-time-mask queued wrapper after the first dev
  follow-up launch exited before evaluation with `env: 'tedlium': No such file
  or directory`. The wrapper now quotes its environment array entries and passes
  `SPLIT` / multi-word `DATASETS` explicitly to the launcher.
- ROB-94: Added a separate fixed-setting encoder-decoder thesis row path under
  `lcasr/results/enc_dec/rob94_fixed_setting_thesis/`. The wrapper queues the
  runnable missing `teacher_ce`, `lr=1e-7`, `freq3_width24_time0`,
  `beam5_lp0p5`, epoch-1 rows and the aggregator keeps the 24-row table
  explicit, marking Rev16 dev unavailable because the current runner exposes
  Rev16 test only.
- ROB-63: Final encoder-decoder result layout has exactly three direct
  checkpoint-family folders under `lcasr/results/enc_dec/`: `enc_dec_v2/`,
  `old_seed/`, and `rl_step_30000/`. The top-level `README.md` and
  `OUTCOME.md` name the exact checkpoint paths; ROB-63 paired seed/RL views
  are regenerated from `rl_step_30000/rob63_rl_self_training_compare/`.
- ROB-63: Added the thesis-facing matched `teacher_ce`, `lr=1e-7`,
  `freq3_width24_time0` comparison to `enc_dec_v2/OUTCOME.md` and the shared
  89.3M-parameter `EncDecSconformerV2` architecture summary to
  `lcasr/results/enc_dec/README.md`. `enc_dec_v2` has matching TED-LIUM and
  Earnings22 rows only; CHIME-6 and Rev16 remain available for `old_seed` and
  `rl_step_30000`, not the historical checkpoint.

## 2026-05-21

- ROB-103: The time-mask-only scaled RMM follow-up completed under `lcasr/results/rmm_eval/ctc_seq16384_time_only_scaled_time_masks/` with 42 PKLs and regenerated `summary.csv`, `summary_by_setting.csv`, and `summary.md`. Each dataset/split/epoch group has `N=3`; test mean WERs are TEDLIUM 5.84%/5.82%, Earnings22 15.62%/15.37%, CHiME-6 79.79%/92.94%, and Rev16 14.13%/14.07% for epochs 1/5 respectively. Dev mean WERs are TEDLIUM 6.72%/6.59%, Earnings22 19.96%/19.79%, and CHiME-6 84.70%/84.45%.

## 2026-05-20

- ROB-68: The scaled-time-mask 16384-context RMM dev follow-up completed for TEDLIUM, Earnings22, and CHiME-6 with 3 repeats per dataset/epoch setting. Regenerated `lcasr/results/rmm_eval/ctc_seq16384_scaled_time_masks/summary.csv`, `summary_by_setting.csv`, and `summary.md` from 42 PKLs total; dev mean WERs are TEDLIUM 6.63%/6.47%, Earnings22 19.98%/18.85%, and CHiME-6 59.07%/57.90% for epochs 1/5 respectively.
- ROB-68: Folded RMM artifacts under `lcasr/results/rmm_eval/` with subdirectories for `ctc_seq2048/`, `ctc_seq16384/`, and `ctc_seq16384_scaled_time_masks/`.

## 2026-05-16

- ROB-68: The scaled-time-mask 16384-context RMM follow-up completed all 24 requested cells under `lcasr/results/rmm_eval/ctc_seq16384_scaled_time_masks/` and regenerated `summary.csv`, `summary_by_setting.csv`, and `summary.md`. Each dataset/epoch group has `N=3`; mean WERs are TEDLIUM 5.84%/5.80%, Earnings22 15.62%/14.73%, CHiME-6 77.92%/77.58%, and Rev16 14.09%/13.90% for epochs 1/5 respectively.

## 2026-05-15

- ROB-68: The fixed-12-mask 16384-context RMM run completed with 3 repeats under `lcasr/results/rmm_eval/ctc_seq16384/`. CHiME-6 remained a real scored deletion result at 100% WER, not an aggregation failure.
- ROB-68: Added the scaled-time-mask RMM variant so 16384-context runs preserve 2048-like per-mask widths by using `96` time masks at `seq_len=16384`.

## 2026-05-14

- ROB-68: A human review clarified that the requested comparison is the normal `seq_len=16384`, `overlap=14336`, `lr=9e-5` setup, not the earlier 2048-context run. Cancelled the queued 2048-context `9e-5` follow-up before it acquired a GPU and added a separate callback-backed 16384-context RMM eval scaffold under `lcasr/results/rmm_eval/ctc_seq16384/` using checkpoint `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_16384_rp_1/step_105360.pt`.
- ROB-51: Added a combined progressive-bottom CTC-head comparison figure target. It plots the completed `progressive_bottom` and `progressive_bottom_ctc_decoder` 9e-5 result families side by side for each bottom-prefix setting, so the effect of always training the CTC decoder can be read directly from a single chart.
- ROB-51: Adjusted the combined progressive-bottom CTC-head comparison figure layout so the legend no longer overlaps the unadapted WER annotation.

## 2026-05-13

- ROB-84: Updated the repo-local Symphony guidance so future Mimas work must
  avoid `/tmp` for scratch, caches, logs, intermediate files, and experiment
  artifacts, using repo-local ignored directories or result-scoped paths
  instead.

- ROB-63: Earnings22 beam5/lp0.5 sanity check confirmed the top-level 28%
  versus ROB-63 25% mismatch is a checkpoint-family difference:
  `enc_dec_v2/step_105360.pt` gives `0.28724` WER, while `old_seed`
  `step_210720.pt` gives `0.25172`. Artifacts live under
  `lcasr/results/enc_dec/enc_dec_v2/rob63_earnings_unadapted_sanity/`.

- ROB-63: Follow-up sweeps are summarized in
  `lcasr/results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/COMBINED_OUTCOME.md`.
  Durable read: CHiME-6 stays deletion-collapsed under these enc-dec
  self-training settings; Rev16 has some lower-LR RL wins but high
  augmentation can make RL deletion-heavy; Earnings22 targeted high-aug
  slightly favors RL over seed but not over RL's own unadapted baseline.

## 2026-05-11

- ROB-68: Added an opt-in RMM random mixed-mask augmentation policy to the CTC dynamic-eval self-training path and scaffolded an all-dataset 2048-context RMM evaluation for epochs `1` and `5`. The run uses checkpoint `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt`, `seq_len=2048`, `overlap=1792`, `lr=1e-5`, and writes under `lcasr/results/rmm_eval/ctc_seq2048/`.
- ROB-68: The callback-backed full RMM eval completed all 8 requested cells. Regenerated `lcasr/results/rmm_eval/ctc_seq2048/summary.csv`, `summary_by_setting.csv`, and `summary.md`; final WERs were TEDLIUM 5.96%/5.83%, Earnings22 15.72%/14.72%, CHiME-6 78.06%/78.71%, and Rev16 14.11%/20.52% for epochs 1/5 respectively.
- ROB-63: Main one-epoch encoder-decoder self-training comparison completed
  for old seed versus RL `step_30000`, with normal baselines and combined
  reporting under `lcasr/results/enc_dec/rl_step_30000/`. Initial TED-LIUM
  and Earnings22 results favored RL in most matched cells, while later CHiME-6
  and Rev16 follow-ups require reading alongside each checkpoint's unadapted
  baseline rather than as standalone adapted WER.
- ROB-66: Added the no-adapt 2048-context CTC baseline scaffold for the ROB-56 checkpoint. The queued wrapper uses `epochs=0`, `seq_len=2048`, `overlap=1792`, test splits for `earnings22`, `tedlium`, `chime6`, and `rev16`, and regenerates thesis-friendly `summary.csv` / `summary.md` under `lcasr/results/ctc_seq2048_unadapted_baseline/`.

## 2026-05-10

- Aggregated the ROB-56 TEDLIUM lower-LR pilot and updated the aggregator so grouped summaries include LR. An initial final pass used epoch-specific LRs, but a later human clarification requested one LR for both epoch counts; under that rule, `lr=1e-5` has the best mean TEDLIUM WER across the epoch-1 and epoch-5 pilot rows and is the shared LR for the replacement final all-dataset pass.
- The ROB-56 shared-LR final all-dataset pass completed successfully for `earnings22`, `tedlium`, `chime6`, and `rev16` on the test split with `seq_len=2048`, `overlap=1792`, adaptation epochs `1` and `5`, and `lr=1e-5`. Recomputed `lcasr/results/ctc_seq2048_self_training_final_lr/summary.csv`, `summary_by_setting.csv`, and `summary.md`; final WER ranges from `5.86%` on TEDLIUM epoch 5 to `61.96%` on CHiME-6 epoch 1.
- ROB-57 entropy ablation: inspected the completed 5-epoch TED-LIUM/Earnings22 test run, regenerated aggregates and plots from the raw traces, and recorded the final WER/entropy summary in `lcasr/results/entropy_ablation/OUTCOME.md`.
- ROB-61: Added a normal encoder-decoder checkpoint benchmark scaffold for the old seed checkpoint versus selected ROB-26 RL checkpoints on TEDLIUM and Earnings22. Copied the old `step_210720.pt` and RL steps `2000`, `10000`, `20000`, and `30000` from Stanage to Mimas under `/store/store5/data/acp21rjf_checkpoints/lcasr/`. The benchmark uses greedy and `beam5_lp0p5` decoding only; no self-training or adaptation.
- ROB-61: The queued benchmark completed all 20 expected normal-eval PKLs and produced `summary.csv` plus `OUTCOME.md`. The detached callback failed because generated log lines made the Linear comment exceed the body size limit, so the callback helper now caps log excerpts and final comment bodies by characters.

## 2026-05-09

- Prepared ROB-56 CTC self-training evaluation for the 1-epoch 2048-context checkpoint `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt`. The queued setup uses `seq_len=2048`, `overlap=1792`, `lr=9e-5`, frequency-only main augmentation (`6` masks, width `34`, no time masks), test splits for `earnings22`, `tedlium`, `chime6`, and `rev16`, and adaptation epochs `1` and `5`. The callback wrapper was smoke-tested in dry-run mode before queueing.
- Patched `scripts/linear_experiment_callback.py` to cap log excerpts and total Linear comment size after the ROB-56 completion callback hit Linear's 100K comment limit on a 139 MB wrapper log with long transcript lines. Future detached callbacks should use this helper rather than embedding uncapped log tails.
- Added a callback-backed ROB-56 lower-LR pilot wrapper for TEDLIUM with `LRS="3e-5 1e-5 3e-6 1e-6"` across adaptation epochs `1` and `5`, using the same 2048-context checkpoint, overlap, and frequency-only augmentation settings as the initial sweep.
- ROB-51: Added a separate `progressive_bottom_ctc_decoder` evaluation family for the follow-up request where the bottom-prefix subset is trainable and the CTC decoder/output projection is always trainable. This writes fresh artifacts under `lcasr/results/ctc_self_training_extra_ablation_sweeps/progressive_bottom_ctc_decoder/` and has its own callback-backed wrapper plus figure target, rather than reusing the completed frozen-head `progressive_bottom` outputs.
- ROB-51: Clarified the progressive ablation plot labels so `progressive_top` labels identify frozen bottom-prefix masks (`freeze <=L0..L5`) while `progressive_bottom` labels identify trainable bottom-prefix masks (`train sub. only`, `train <=L0..L5`). The underlying result artifacts are unchanged; this only makes the thesis-facing figures less ambiguous.
- ROB-51: The corrected `progressive_bottom` bottom-prefix training run completed with 28 fresh PKLs: Earnings22 test for `9e-6`, `9e-5`, and `9e-4`, plus TED-LIUM test for `9e-5`. The `9e-5` results move in the expected direction as more bottom layers are trainable: TED-LIUM improves from 6.38% WER with subsampling-only training to 5.99% through layer 5, and Earnings22 improves from 18.54% to 16.03%. This is the intended opposite pattern to progressively freezing the lower stack in `progressive_top`.
- ROB-57 entropy ablation: added an opt-in CTC entropy trace for dynamic evaluation, plus aggregation/plotting scaffolding and queued-run wrapper support. The Linear-approved plan was constrained by the latest human comment to record 5 total adaptation epochs.

## 2026-05-08

- ROB-51: Added a fresh `progressive_bottom` CTC self-training ablation path. This is a separate evaluation family that writes new outputs under `lcasr/results/ctc_self_training_extra_ablation_sweeps/progressive_bottom/`; existing `progressive_top` PKLs must not be reused for this issue.
- ROB-51: The fresh `progressive_bottom` run completed and wrote 32 PKLs plus refreshed aggregate tables and `progressive_bottom_ablation_bars.pdf`. The detached wrapper's Linear callback initially failed because the helper path was relative to the final analysis directory; the wrapper now calls the callback helper by absolute repo path and supports callback-only smoke checks.
- ROB-51: Reworked `progressive_bottom` after review clarified the intended mask: train only subsampling, then train only subsampling plus layers.0..N. The previous `--freeze_subsampling --freeze_layers_through N` artifacts froze that bottom prefix instead and must be replaced by a fresh bottom-prefix training run.
