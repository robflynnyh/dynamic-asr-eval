# CTC 16384-Context RMM Self-Training Eval

ROB-68 evaluates the normal 16384-context CTC dynamic-evaluation setup with the RMM random mixed-mask augmentation policy ported from `learning-to-augment`.

## Setup

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_16384_rp_1/step_105360.pt`
- Sequence length: `16384`
- Overlap: `14336`
- Datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Split: `test`
- Adaptation epochs: `1`, `5`
- LR: `9e-5`
- Augmentation: `augmentation_policy='rmm'`
- RMM policy: random time-only, frequency-only, or time+frequency masks; `time_masks=12`, `freq_masks=5..7`, `freq_mask_param=24..44`, `zero_masking=True`
- Repeats: `1`

The earlier `lcasr/results/ctc_seq2048_rmm_eval/` artifacts are a 2048-context comparison and should not be used as the normal 16384-context ROB-68 result.

## Launch

From `lcasr/`:

```bash
DATASETS="earnings22 tedlium chime6 rev16" EPOCHS="1 5" REPEATS=1 \
  bash launch_scripts/run_ctc_seq16384_rmm_eval.sh
```

ROB-68 uses the callback-backed detached wrapper from the repo root:

```bash
screen -L -Logfile lcasr/results/ctc_seq16384_rmm_eval/logs/rob68_ctc_seq16384_rmm_eval.screen.log \
  -dmS rob68_ctc_seq16384_rmm_eval \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-68 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob68_ctc_seq16384_rmm_queued.sh'
```

## Outputs

Each run writes:

```text
<dataset>-test-ctc-seq16384-overlap14336-rmm-epoch-<epoch>-lr-9em5_<repeat>.pkl
```

Run aggregation after the callback completes:

```bash
python lcasr/results/ctc_seq16384_rmm_eval/aggregate.py
```

## Completed Results

The completed callback-backed run exited with status `0` and produced all 8
expected PKLs. Aggregated WERs:

| Dataset | Epoch 1 WER | Epoch 5 WER |
|---|---:|---:|
| TEDLIUM | 5.93% | 5.82% |
| Earnings22 | 15.76% | 15.59% |
| CHiME-6 | 100.00% | 100.00% |
| Rev16 | 14.22% | 14.09% |

CHiME-6 scored as 100% deletion because both long-recording PKLs contain empty
normalized `model_output` strings against non-empty references. This is a real
scored run result, not an aggregation failure.
