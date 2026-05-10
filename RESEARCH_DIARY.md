# Research Diary

## 2026-05-10

- ROB-57 entropy ablation: inspected the completed 5-epoch TED-LIUM/Earnings22 test run, regenerated aggregates and plots from the raw traces, and recorded the final WER/entropy summary in `lcasr/results/entropy_ablation/OUTCOME.md`.

## 2026-05-09

- ROB-51: Added a separate `progressive_bottom_ctc_decoder` evaluation family for the follow-up request where the bottom-prefix subset is trainable and the CTC decoder/output projection is always trainable. This writes fresh artifacts under `lcasr/results/ctc_self_training_extra_ablation_sweeps/progressive_bottom_ctc_decoder/` and has its own callback-backed wrapper plus figure target, rather than reusing the completed frozen-head `progressive_bottom` outputs.
- ROB-51: Clarified the progressive ablation plot labels so `progressive_top` labels identify frozen bottom-prefix masks (`freeze <=L0..L5`) while `progressive_bottom` labels identify trainable bottom-prefix masks (`train sub. only`, `train <=L0..L5`). The underlying result artifacts are unchanged; this only makes the thesis-facing figures less ambiguous.
- ROB-51: The corrected `progressive_bottom` bottom-prefix training run completed with 28 fresh PKLs: Earnings22 test for `9e-6`, `9e-5`, and `9e-4`, plus TED-LIUM test for `9e-5`. The `9e-5` results move in the expected direction as more bottom layers are trainable: TED-LIUM improves from 6.38% WER with subsampling-only training to 5.99% through layer 5, and Earnings22 improves from 18.54% to 16.03%. This is the intended opposite pattern to progressively freezing the lower stack in `progressive_top`.
- ROB-57 entropy ablation: added an opt-in CTC entropy trace for dynamic evaluation, plus aggregation/plotting scaffolding and queued-run wrapper support. The Linear-approved plan was constrained by the latest human comment to record 5 total adaptation epochs.

## 2026-05-08

- ROB-51: Added a fresh `progressive_bottom` CTC self-training ablation path. This is a separate evaluation family that writes new outputs under `lcasr/results/ctc_self_training_extra_ablation_sweeps/progressive_bottom/`; existing `progressive_top` PKLs must not be reused for this issue.
- ROB-51: The fresh `progressive_bottom` run completed and wrote 32 PKLs plus refreshed aggregate tables and `progressive_bottom_ablation_bars.pdf`. The detached wrapper's Linear callback initially failed because the helper path was relative to the final analysis directory; the wrapper now calls the callback helper by absolute repo path and supports callback-only smoke checks.
- ROB-51: Reworked `progressive_bottom` after review clarified the intended mask: train only subsampling, then train only subsampling plus layers.0..N. The previous `--freeze_subsampling --freeze_layers_through N` artifacts froze that bottom prefix instead and must be replaced by a fresh bottom-prefix training run.
