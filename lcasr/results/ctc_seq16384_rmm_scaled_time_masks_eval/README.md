# CTC 16384-Context RMM Scaled-Time-Mask Eval

ROB-68 follow-up for the human comment on 2026-05-15 requesting 2048-like
time-mask widths at the normal 16384-context setup.

## Setup

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_16384_rp_1/step_105360.pt`
- Sequence length: `16384`
- Overlap: `14336`
- Test datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Dev datasets: `earnings22`, `tedlium`, `chime6`; `rev16` is test-only in
  `run_dynamic_eval_full.py`
- Splits: `test` plus a dev follow-up for `earnings22`, `tedlium`, and
  `chime6`; `rev16` has no dev split in this runner
- Adaptation epochs: `1`, `5`
- LR: `9e-5`
- Augmentation: `augmentation_policy='rmm'`
- RMM time masks: `rmm_scale_time_masks_by_seq_len=True`, reference sequence length `2048`
- Effective time-mask count at `seq_len=16384`: `96`, preserving the per-mask width implied by the 2048-context RMM default of `12` masks
- RMM frequency masks: unchanged from the original port, `freq_masks=5..7`, `freq_mask_param=24..44`, `zero_masking=True`
- Repeats: `3`

This directory is intentionally separate from
`lcasr/results/ctc_seq16384_rmm_eval/`, which contains the completed fixed
`time_masks=12` 16384-context run.

## Launch

The callback-backed detached wrapper is launched from the repo root:

```bash
screen -L -Logfile lcasr/results/ctc_seq16384_rmm_scaled_time_masks_eval/logs/rob68_ctc_seq16384_rmm_scaled_time_masks.screen.log \
  -dmS rob68_ctc_seq16384_rmm_scaled_time_masks \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-68 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- env SCREEN_NAME=rob68_ctc_seq16384_rmm_scaled_time_masks bash scripts/run_rob68_ctc_seq16384_rmm_scaled_time_queued.sh'
```

The dev follow-up uses the same wrapper with `SPLIT=dev` and no `rev16`:

```bash
screen -L -Logfile lcasr/results/ctc_seq16384_rmm_scaled_time_masks_eval/logs/rob68_ctc_seq16384_rmm_scaled_time_masks_dev.screen.log \
  -dmS rob68_ctc_seq16384_rmm_scaled_time_masks_dev \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-68 && SCREEN_NAME=rob68_ctc_seq16384_rmm_scaled_time_masks_dev SPLIT=dev DATASETS="earnings22 tedlium chime6" CALLBACK_TARGET_STATE=Todo /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob68_ctc_seq16384_rmm_scaled_time_queued.sh'
```

Run aggregation after the callback completes:

```bash
python lcasr/results/ctc_seq16384_rmm_scaled_time_masks_eval/aggregate.py
```

## Outputs

Each run writes:

```text
<dataset>-<split>-ctc-seq16384-overlap14336-rmm-width2048-scaled-epoch-<epoch>-lr-9em5_<repeat>.pkl
```

## Results

The test run completed on 2026-05-16. The dev follow-up completed on
2026-05-20. `summary.csv`, `summary_by_setting.csv`, and `summary.md` are
regenerated from all available PKLs in this directory.

| Dataset | Split | Epoch 1 | Epoch 5 |
|---|---|---:|---:|
| TEDLIUM | dev | 6.63% | 6.47% |
| TEDLIUM | test | 5.84% | 5.80% |
| Earnings22 | dev | 19.98% | 18.85% |
| Earnings22 | test | 15.62% | 14.73% |
| CHiME-6 | dev | 59.07% | 57.90% |
| CHiME-6 | test | 77.92% | 77.58% |
| Rev16 | test | 14.09% | 13.90% |
