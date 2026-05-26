# ROB-122 ROB-81 Floras-Finetuned Encoder-Decoder Eval

This folder tracks the ROB-122 evaluation of the ROB-81 supervised Floras-50
finetuned encoder-decoder checkpoint:

```text
/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0/step_323484.pt
```

## Target Rows

The run uses the same thesis-facing setting as the existing ROB-94 fixed
encoder-decoder rows:

| Row type | Setting |
|---|---|
| Unadapted baseline | `epochs=0`, `beam5_lp0p5`, `seq=2048`, `overlap=0`, no augmentation |
| Adaptation | `teacher_ce`, `epochs=1`, `lr=1e-7`, `freq3_width24_time0`, `beam5_lp0p5`, `seq=2048`, `overlap=0` |

Datasets and splits:

| Dataset | Splits |
|---|---|
| `tedlium` | `dev`, `test` |
| `earnings22` | `dev`, `test` |
| `chime6` | `dev`, `test` |
| `rev16` | `test` |

Rev16 dev is intentionally unavailable because the current `lcasr/rev16`
loader exposes only the test split.

## Launch

From the repository root, after validation:

```bash
screen -L -Logfile lcasr/results/enc_dec/rob81_floras50_finetune_eval/screen.log \
  -dmS rob122_rob81_floras_eval \
  bash -lc '/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob122_rob81_floras_eval_queued.sh'
```

The queued wrapper posts a Linear callback to ROB-122 on any exit path and
moves the issue back to `Todo` for result inspection.

## Regeneration

From `lcasr/`:

```bash
python results/enc_dec/rob81_floras50_finetune_eval/aggregate.py \
  --csv results/enc_dec/rob81_floras50_finetune_eval/summary.csv \
  --outcome results/enc_dec/rob81_floras50_finetune_eval/OUTCOME.md
```
