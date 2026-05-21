# CTC 65536-Context Higher-LR Self-Training Follow-Up

ROB-67 follow-up sweep requested after the completed `lr=1e-5` 65536-context adapted evaluation. This keeps higher-LR rows separate from the baseline ROB-67 result directory.

- Stanage checkpoint: `/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_65536_rp_1/step_105360.pt`
- Sequence length: `65536`
- Overlap: `57344`
- Datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Split: `test`
- Adaptation epochs: `1`, `5`
- LRs: `9e-5`, `3e-4`
- Augmentation: `spec_augment_n_freq_masks=6`, `spec_augment_freq_mask_param=34`, `spec_augment_n_time_masks=0`
- Repeats: `1`

Submit from the Stanage checkout:

```bash
bash scripts/submit_rob67_ctc_seq65536_higher_lr_stanage.sh
```

Aggregate after completion:

```bash
python lcasr/results/seq_65536_investigation/aggregate_adapted.py \
  --root lcasr/results/seq_65536_investigation/self_training_higher_lr
```

## Completed Stanage Run

- Array job: `10169175`
- Finalizer job: `10169176`
- Finalizer log: `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-67/higher-lr-finalize-10169176.log`
- Stanage result root: `/mnt/parscratch/users/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-67/lcasr/results/seq_65536_investigation/self_training_higher_lr`
- Completion status: all `16/16` expected PKLs present, with regenerated `summary.csv`, `summary_by_setting.csv`, and `summary.md`.

Best higher-LR WER by dataset:

| Dataset | Best setting | WER |
|---|---:|---:|
| TEDLIUM | epoch 5, `lr=9e-5` | 5.81% |
| Earnings22 | epoch 5, `lr=3e-4` | 15.01% |
| CHiME-6 | epoch 5, `lr=9e-5` | 75.17% |
| Rev16 | epoch 5, `lr=3e-4` | 14.11% |

Compared with the earlier `lr=1e-5` ROB-67 adapted run, these higher-LR rows improve the best epoch-5 TEDLIUM, Earnings22, and Rev16 WERs. CHiME-6 is the exception: the earlier `lr=1e-5`, epoch-5 row remains better at 74.57% WER.
