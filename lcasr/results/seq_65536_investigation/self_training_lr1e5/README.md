# CTC 65536-Context Self-Training Eval

ROB-67 evaluates the CTC model trained with `sequence_scheduler.max_sequence_length=65536` using the same dynamic-evaluation self-training protocol as the ROB-56 2048-context final pass.

- Mimas checkpoint: `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_65536_rp_1/step_105360.pt`
- Stanage checkpoint: `/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_65536_rp_1/step_105360.pt`
- Sequence length: `65536`
- Overlap: `57344` (`65536 * 0.875`)
- Datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Split: `test`
- Adaptation epochs: `1`, `5`
- LR: `1e-5`
- Augmentation: `spec_augment_n_freq_masks=6`, `spec_augment_freq_mask_param=34`, `spec_augment_n_time_masks=0`
- Repeats: `1`

Manual launch from `lcasr/`:

```bash
DATASETS="earnings22 tedlium chime6 rev16" EPOCHS="1 5" REPEATS=1 \
  bash launch_scripts/run_ctc_seq65536_self_training_eval.sh
```

ROB-67 is intended to run on Stanage because Mimas is unlikely to have enough GPU memory for self-training at this context length. Use `scripts/submit_rob67_ctc_seq65536_stanage.sh` from a Stanage checkout after the CPU smoke passes.

Aggregate after completion:

```bash
python lcasr/results/seq_65536_investigation/aggregate_adapted.py
```
