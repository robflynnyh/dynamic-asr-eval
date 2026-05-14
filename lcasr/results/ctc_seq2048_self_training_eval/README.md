# CTC 2048-Context Self-Training Eval

ROB-56 evaluates the 1-epoch CTC model trained with `sequence_scheduler.max_sequence_length=2048` on the current dynamic-evaluation self-training protocol.

## Setup

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt`
- Sequence length: `2048`
- Overlap: `1792` (`2048 * 0.875`, matching the established long-context overlap rule)
- Datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Split: `test`
- Adaptation epochs: `1`, `5`
- LR: `9e-5`
- Augmentation: `spec_augment_n_freq_masks=6`, `spec_augment_freq_mask_param=34`, `spec_augment_n_time_masks=0`
- Repeats: `1`

## Launch

From `lcasr/`:

```bash
DATASETS="earnings22 tedlium chime6 rev16" EPOCHS="1 5" REPEATS=1 \
  bash launch_scripts/run_ctc_seq2048_self_training_eval.sh
```

ROB-56 uses the callback-backed detached wrapper from the repo root:

```bash
screen -L -Logfile lcasr/results/ctc_seq2048_self_training_eval/logs/rob56_ctc_seq2048_self_training.screen.log \
  -dmS rob56_ctc_seq2048_self_training \
  bash -lc '/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob56_ctc_seq2048_self_training_queued.sh'
```

## Outputs

Each run writes:

```text
<dataset>-test-ctc-seq2048-overlap1792-epoch-<epoch>-lr-9em5_<repeat>.pkl
```

Run aggregation after the callback completes:

```bash
python lcasr/results/ctc_seq2048_self_training_eval/aggregate.py
```
