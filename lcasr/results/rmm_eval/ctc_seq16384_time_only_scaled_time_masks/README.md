# CTC 16384-Context Time-Mask-Only Scaled RMM Eval

ROB-103 follow-up from ROB-68 evaluating the scaled RMM time-mask component
without sampling the RMM frequency-only or time+frequency branches.

## Setup

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_16384_rp_1/step_105360.pt`
- Sequence length: `16384`
- Overlap: `14336`
- Test datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Dev datasets: `earnings22`, `tedlium`, `chime6`; `rev16` is test-only in
  `run_dynamic_eval_full.py`
- Adaptation epochs: `1`, `5`
- LR: `9e-5`
- Augmentation: `augmentation_policy='rmm'`, `rmm_branch='time'`
- RMM time masks: `rmm_scale_time_masks_by_seq_len=True`, reference sequence length `2048`
- Effective time-mask count at `seq_len=16384`: `96`, preserving the per-mask width implied by the 2048-context RMM default of `12` masks
- RMM frequency masks: configured as in RMM but not selected because `rmm_branch='time'`
- Repeats: `3`

This directory is intentionally separate from
`lcasr/results/rmm_eval/ctc_seq16384_scaled_time_masks/`, which contains the
ROB-68 scaled mixed-branch RMM result.

## Launch

The callback-backed detached wrapper is launched from the repo root:

```bash
screen -L -Logfile lcasr/results/rmm_eval/ctc_seq16384_time_only_scaled_time_masks/logs/rob103_ctc_seq16384_time_only_scaled_rmm.screen.log \
  -dmS rob103_ctc_seq16384_time_only_scaled_rmm \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-103 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob103_ctc_seq16384_time_only_scaled_rmm_queued.sh'
```

The wrapper runs the ROB-68 scaled-time-mask result scope by default: test for
`earnings22`, `tedlium`, `chime6`, and `rev16`, then dev for `earnings22`,
`tedlium`, and `chime6`. To run only one split, set `RUN_TEST=0` or
`RUN_DEV=0`.

Run aggregation after the callback completes:

```bash
python lcasr/results/rmm_eval/ctc_seq16384_time_only_scaled_time_masks/aggregate.py
```

## Outputs

Each run writes:

```text
<dataset>-<split>-ctc-seq16384-overlap14336-rmm-time-only-width2048-scaled-epoch-<epoch>-lr-9em5_<repeat>.pkl
```

The aggregate script writes `summary.csv`, `summary_by_setting.csv`, and
`summary.md` from the available PKLs.
