# CTC 65536-Context Longer-Epoch Follow-Up

ROB-67 follow-up requested after reviewing the completed 65536-context result table: run the adapted CTC evaluation for 10 and 20 self-training epochs at `lr=9e-5` and `lr=3e-4`.

- Stanage checkpoint: `/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_65536_rp_1/step_105360.pt`
- Sequence length: `65536`
- Overlap: `57344`
- Datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Split: `test`
- Adaptation epochs: `10`, `20`
- LRs: `9e-5`, `3e-4`
- Augmentation: `spec_augment_n_freq_masks=6`, `spec_augment_freq_mask_param=34`, `spec_augment_n_time_masks=0`
- Repeats: `1`

Submit from the Stanage checkout:

```bash
bash scripts/submit_rob67_ctc_seq65536_longer_epochs_stanage.sh
```

Aggregate after completion:

```bash
python lcasr/results/ctc_seq65536_self_training_eval/aggregate.py \
  --root lcasr/results/ctc_seq65536_self_training_longer_epochs
```
