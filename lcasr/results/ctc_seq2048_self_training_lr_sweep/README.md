# CTC 2048-Context Self-Training Lower-LR Pilot

ROB-56 lower-LR pilot after the initial `9e-5` sweep. The pilot keeps the same
2048-context checkpoint and self-training settings but runs one dataset first to
choose a better LR before a possible all-dataset final pass.

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt`
- Dataset: `tedlium`
- Split: `test`
- Sequence length: `2048`
- Overlap: `1792`
- Adaptation epochs: `1`, `5`
- LRs: `3e-5`, `1e-5`, `3e-6`, `1e-6`
- Augmentation: `spec_augment_n_freq_masks=6`, `spec_augment_freq_mask_param=34`, `spec_augment_n_time_masks=0`
- Repeats: `1`

Launch from the repo root with the callback-backed wrapper:

```bash
screen -L -Logfile lcasr/results/ctc_seq2048_self_training_lr_sweep/logs/rob56_ctc_seq2048_lr_sweep_tedlium.screen.log \
  -dmS rob56_ctc_seq2048_lr_sweep_tedlium \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-56 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob56_ctc_seq2048_lr_sweep_queued.sh'
```

After the callback completes, aggregate with:

```bash
python lcasr/results/ctc_seq2048_self_training_eval/aggregate.py \
  --root lcasr/results/ctc_seq2048_self_training_lr_sweep
```
