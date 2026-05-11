# CTC 2048-Context RMM Self-Training Eval

ROB-68 evaluates the current default 2048-context CTC dynamic-evaluation setup with the RMM random mixed-mask augmentation policy ported from `learning-to-augment`.

## Setup

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt`
- Sequence length: `2048`
- Overlap: `1792`
- Datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Split: `test`
- Adaptation epochs: `1`, `5`
- LR: `1e-5`
- Augmentation: `augmentation_policy='rmm'`
- RMM policy: random time-only, frequency-only, or time+frequency masks; `time_masks=12`, `freq_masks=5..7`, `freq_mask_param=24..44`, `zero_masking=True`
- Repeats: `1`

## Launch

From `lcasr/`:

```bash
DATASETS="earnings22 tedlium chime6 rev16" EPOCHS="1 5" REPEATS=1 \
  bash launch_scripts/run_ctc_seq2048_rmm_eval.sh
```

ROB-68 uses the callback-backed detached wrapper from the repo root:

```bash
screen -L -Logfile lcasr/results/ctc_seq2048_rmm_eval/logs/rob68_ctc_seq2048_rmm_eval.screen.log \
  -dmS rob68_ctc_seq2048_rmm_eval \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-68 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob68_ctc_seq2048_rmm_queued.sh'
```

## Outputs

Each run writes:

```text
<dataset>-test-ctc-seq2048-overlap1792-rmm-epoch-<epoch>-lr-1em5_<repeat>.pkl
```

Run aggregation after the callback completes:

```bash
python lcasr/results/ctc_seq2048_rmm_eval/aggregate.py
```
