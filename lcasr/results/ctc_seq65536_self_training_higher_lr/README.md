# CTC 65536-Context Higher-LR Self-Training Follow-Up

ROB-67 follow-up sweep requested after the completed `lr=1e-5` 65536-context adapted evaluation. This keeps higher-LR rows separate from the baseline ROB-67 result directory.

- Stanage checkpoint: `/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_65536_rp_1/step_105360.pt`
- Sequence length: `65536`
- Overlap: `57344`
- Datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Split: `test`
- Adaptation epochs: `1`, `5`
- LRs: `3e-5`, `9e-5`
- Augmentation: `spec_augment_n_freq_masks=6`, `spec_augment_freq_mask_param=34`, `spec_augment_n_time_masks=0`
- Repeats: `1`

Submit from the Stanage checkout:

```bash
bash scripts/submit_rob67_ctc_seq65536_higher_lr_stanage.sh
```

Aggregate after completion:

```bash
python lcasr/results/ctc_seq65536_self_training_eval/aggregate.py \
  --root lcasr/results/ctc_seq65536_self_training_higher_lr
```
