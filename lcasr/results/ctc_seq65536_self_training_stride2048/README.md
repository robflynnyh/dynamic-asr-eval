# CTC 65536-Context Stride-2048 Self-Training Follow-Up

ROB-67 follow-up requested after reviewing PR #18: rerun the adapted 65536-context CTC evaluation at `lr=9e-5` for 5 self-training epochs, but use the same stride as the normal 16384 setup so the number of adaptation windows is comparable.

- Stanage checkpoint: `/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_65536_rp_1/step_105360.pt`
- Sequence length: `65536`
- Overlap: `63488`
- Stride: `2048`
- Datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Split: `test`
- Adaptation epochs: `5`
- LR: `9e-5`
- Augmentation: `spec_augment_n_freq_masks=6`, `spec_augment_freq_mask_param=34`, `spec_augment_n_time_masks=0`
- Repeats: `1`
- Expected adapted outputs: `4` PKLs

The standard 65536-context ROB-67 runs used `overlap=57344`, giving stride `8192`. The normal 16384 setup uses `overlap=14336`, giving stride `2048`; this follow-up therefore keeps `seq_len=65536` and sets `overlap=63488`.

Submit from the Stanage checkout after the CPU smoke passes:

```bash
bash scripts/submit_rob67_ctc_seq65536_stride2048_stanage.sh
```

The dependent finalizer reuses `scripts/run_rob67_ctc_seq65536_higher_lr_finalize.sbatch`
with `FINALIZER_KIND=stride2048`, so the callback note and queued command are
constructed inside the finalizer rather than passed as long `sbatch --export`
values.

Aggregate after completion:

```bash
python lcasr/results/ctc_seq65536_self_training_eval/aggregate.py \
  --root lcasr/results/ctc_seq65536_self_training_stride2048
```
