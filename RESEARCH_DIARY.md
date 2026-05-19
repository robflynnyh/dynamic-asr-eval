# Research Diary

## 2026-05-19

- ROB-68: After a human follow-up requested dev results for the latest
  scaled-time-mask setup, prepared the existing
  `ctc_seq16384_rmm_scaled_time_masks_eval` package for mixed split summaries
  and queued a dev split pass for `earnings22`, `tedlium`, and `chime6`.
  `rev16` is excluded because the CTC dynamic-eval runner asserts that Rev16
  only supports `test`.
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

## 2026-05-16

- ROB-68: The scaled-time-mask 16384-context RMM follow-up completed all 24 requested cells under `lcasr/results/ctc_seq16384_rmm_scaled_time_masks_eval/` and regenerated `summary.csv`, `summary_by_setting.csv`, and `summary.md`. Each dataset/epoch group has `N=3`; mean WERs are TEDLIUM 5.84%/5.80%, Earnings22 15.62%/14.73%, CHiME-6 77.92%/77.58%, and Rev16 14.09%/13.90% for epochs 1/5 respectively.

## 2026-05-15

- ROB-68: The corrected 16384-context RMM eval completed all 8 requested cells and regenerated `lcasr/results/ctc_seq16384_rmm_eval/summary.csv`, `summary_by_setting.csv`, and `summary.md` from the PKLs. Final WERs were TEDLIUM 5.93%/5.82%, Earnings22 15.76%/15.59%, CHiME-6 100.00%/100.00%, and Rev16 14.22%/14.09% for epochs 1/5 respectively. The CHiME-6 PKLs contain empty normalized model outputs for both long recordings, so the 100% WER is a real scored deletion result rather than an aggregation failure.
- ROB-68: A follow-up human comment requested 3 repeats total. Added a repeat-fill mode for the 16384-context RMM launcher plus a callback-backed wrapper that queues only missing repeats `2` and `3` while preserving the completed repeat `1` artifacts.
- ROB-68: The 16384-context RMM repeat-fill run completed repeats `2` and `3` for all datasets and both epoch settings. Regenerated `lcasr/results/ctc_seq16384_rmm_eval/summary.csv`, `summary_by_setting.csv`, and `summary.md` from 24 PKLs; each dataset/epoch group now has `N=3`. Mean WERs are TEDLIUM 5.97%/5.79%, Earnings22 15.84%/15.35%, CHiME-6 100.00%/100.00%, and Rev16 14.21%/14.04% for epochs 1/5 respectively.
- ROB-68: A human follow-up requested preserving 2048-like RMM time-mask widths in the 16384 setup by scaling the number of time masks instead of the width. Added an opt-in `rmm_scale_time_masks_by_seq_len` setting, a separate result package under `lcasr/results/ctc_seq16384_rmm_scaled_time_masks_eval/`, and a callback-backed wrapper for a 3-repeat all-dataset follow-up using `96` time masks at `seq_len=16384`.

## 2026-05-14

- ROB-68: A human review clarified that the requested comparison is the normal `seq_len=16384`, `overlap=14336`, `lr=9e-5` setup, not the earlier 2048-context run. Cancelled the queued 2048-context `9e-5` follow-up before it acquired a GPU and added a separate callback-backed 16384-context RMM eval scaffold under `lcasr/results/ctc_seq16384_rmm_eval/` using checkpoint `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_16384_rp_1/step_105360.pt`.
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

- ROB-68: Added an opt-in RMM random mixed-mask augmentation policy to the CTC dynamic-eval self-training path and scaffolded an all-dataset 2048-context RMM evaluation for epochs `1` and `5`. The run uses checkpoint `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt`, `seq_len=2048`, `overlap=1792`, `lr=1e-5`, and writes under `lcasr/results/ctc_seq2048_rmm_eval/`.
- ROB-68: The callback-backed full RMM eval completed all 8 requested cells. Regenerated `lcasr/results/ctc_seq2048_rmm_eval/summary.csv`, `summary_by_setting.csv`, and `summary.md`; final WERs were TEDLIUM 5.96%/5.83%, Earnings22 15.72%/14.72%, CHiME-6 78.06%/78.71%, and Rev16 14.11%/20.52% for epochs 1/5 respectively.
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
