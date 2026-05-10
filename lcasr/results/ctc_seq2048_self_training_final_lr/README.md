# CTC 2048-Context Self-Training Selected-LR Eval

ROB-56 final all-dataset pass after the TEDLIUM lower-LR pilot. This keeps the
same 2048-context checkpoint, overlap, and frequency-only main augmentation as
the initial all-dataset sweep, but uses the best single lower learning rate for
both adaptation epoch counts.

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt`
- Datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Split: `test`
- Sequence length: `2048`
- Overlap: `1792`
- LR: `1e-5` for both epoch counts
- Augmentation: `spec_augment_n_freq_masks=6`, `spec_augment_freq_mask_param=34`, `spec_augment_n_time_masks=0`
- Repeats: `1`

The shared LR was selected from
`lcasr/results/ctc_seq2048_self_training_lr_sweep/summary_by_setting.csv`:
averaging TEDLIUM WER across the epoch-1 and epoch-5 pilot rows gives `1e-5`
as the best single LR (`5.929%` mean WER), narrowly ahead of `1e-6`
(`5.944%`) and `3e-6` (`5.945%`).

Launch from the repo root with the callback-backed wrapper:

```bash
screen -L -Logfile lcasr/results/ctc_seq2048_self_training_final_lr/logs/rob56_ctc_seq2048_shared_lr_all_datasets.screen.log \
  -dmS rob56_ctc_seq2048_shared_lr_all_datasets \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-56 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob56_ctc_seq2048_selected_lr_queued.sh'
```

After the callback completes, aggregate with:

```bash
python lcasr/results/ctc_seq2048_self_training_eval/aggregate.py \
  --root lcasr/results/ctc_seq2048_self_training_final_lr
```
