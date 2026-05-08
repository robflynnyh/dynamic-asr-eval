# Research Diary

## 2026-05-08

- ROB-51: Added a fresh `progressive_bottom` CTC self-training ablation path. This is a separate evaluation family that writes new outputs under `lcasr/results/ctc_self_training_extra_ablation_sweeps/progressive_bottom/`; existing `progressive_top` PKLs must not be reused for this issue.
- ROB-51: The fresh `progressive_bottom` run completed and wrote 32 PKLs plus refreshed aggregate tables and `progressive_bottom_ablation_bars.pdf`. The detached wrapper's Linear callback initially failed because the helper path was relative to the final analysis directory; the wrapper now calls the callback helper by absolute repo path and supports callback-only smoke checks.
- ROB-51: Reworked `progressive_bottom` after review clarified the intended mask: train only subsampling, then train only subsampling plus layers.0..N. The previous `--freeze_subsampling --freeze_layers_through N` artifacts froze that bottom prefix instead and must be replaced by a fresh bottom-prefix training run.
