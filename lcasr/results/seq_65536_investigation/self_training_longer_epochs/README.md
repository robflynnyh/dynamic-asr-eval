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
- Stanage array job: `10220452`
- Stanage finalizer job: `10220453`
- Finalizer log: `/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-67/longer-epochs-finalize-10220453.log`

The finalizer reported all 16 expected adapted PKLs. The largest synced PKL is about 1.9 MB, below the 95 MB commit threshold.

## Results

| Dataset | Best longer-epoch setting | WER | Previous best 65536 adapted row | Note |
|---|---:|---:|---:|---|
| TEDLIUM | 20 epochs, `lr=9e-5` | 5.73% | 5 epochs, `lr=9e-5`: 5.81% | Longer adaptation is a small gain. |
| Earnings22 | 20 epochs, `lr=9e-5` | 14.83% | 5 epochs, `lr=3e-4`: 15.01% | Longer adaptation is a small gain. |
| CHiME-6 | 10 epochs, `lr=9e-5` | 74.75% | 5 epochs, `lr=1e-5`: 74.57% | The earlier lower-LR run remains best. The 20-epoch `lr=3e-4` row collapsed to 99.99% WER, almost entirely deletions. |
| Rev16 | 10 epochs, `lr=9e-5` | 14.03% | 5 epochs, `lr=3e-4`: 14.11% | Longer adaptation is a small gain. |

Full per-setting results are in `summary.csv`, `summary_by_setting.csv`, and `summary.md`.

Submit from the Stanage checkout:

```bash
bash scripts/submit_rob67_ctc_seq65536_longer_epochs_stanage.sh
```

Aggregate after completion:

```bash
python lcasr/results/seq_65536_investigation/aggregate_adapted.py \
  --root lcasr/results/seq_65536_investigation/self_training_longer_epochs
```
