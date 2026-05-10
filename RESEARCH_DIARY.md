# Research Diary

## 2026-05-09

- Prepared ROB-56 CTC self-training evaluation for the 1-epoch 2048-context checkpoint `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt`. The queued setup uses `seq_len=2048`, `overlap=1792`, `lr=9e-5`, frequency-only main augmentation (`6` masks, width `34`, no time masks), test splits for `earnings22`, `tedlium`, `chime6`, and `rev16`, and adaptation epochs `1` and `5`. The callback wrapper was smoke-tested in dry-run mode before queueing.
- Patched `scripts/linear_experiment_callback.py` to cap log excerpts and total Linear comment size after the ROB-56 completion callback hit Linear's 100K comment limit on a 139 MB wrapper log with long transcript lines. Future detached callbacks should use this helper rather than embedding uncapped log tails.
- Added a callback-backed ROB-56 lower-LR pilot wrapper for TEDLIUM with `LRS="3e-5 1e-5 3e-6 1e-6"` across adaptation epochs `1` and `5`, using the same 2048-context checkpoint, overlap, and frequency-only augmentation settings as the initial sweep.

## 2026-05-10

- Aggregated the ROB-56 TEDLIUM lower-LR pilot and updated the aggregator so grouped summaries include LR. An initial final pass used epoch-specific LRs, but a later human clarification requested one LR for both epoch counts; under that rule, `lr=1e-5` has the best mean TEDLIUM WER across the epoch-1 and epoch-5 pilot rows and is the shared LR for the replacement final all-dataset pass.
