# ROB-122 ROB-81 Floras-finetuned encoder-decoder eval

Fixed adaptation setting: `teacher_ce`, `lr=1e-7`, `freq3_width24_time0`, `beam5_lp0p5`, epoch `1`, no filtering.
Matched baseline setting: `no_adapt`, `beam5_lp0p5`, epoch `0`, `optim_lr=0.0`, no augmentation.

Completed rows: 0/16. Missing rows: 14. Unavailable rows: 2.

Rev16 dev rows are marked unavailable because the current `lcasr/rev16` loader exposes test only.

## Checkpoint

`/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0/step_323484.pt`

## Rows

| Dataset | Split | Mode | Status | WER | Delta vs no-adapt | Source / note |
|---|---|---|---|---:|---:|---|
| tedlium | dev | `no_adapt` | missing |  |  |  |
| tedlium | dev | `teacher_ce` | missing |  |  |  |
| tedlium | test | `no_adapt` | missing |  |  |  |
| tedlium | test | `teacher_ce` | missing |  |  |  |
| earnings22 | dev | `no_adapt` | missing |  |  |  |
| earnings22 | dev | `teacher_ce` | missing |  |  |  |
| earnings22 | test | `no_adapt` | missing |  |  |  |
| earnings22 | test | `teacher_ce` | missing |  |  |  |
| chime6 | dev | `no_adapt` | missing |  |  |  |
| chime6 | dev | `teacher_ce` | missing |  |  |  |
| chime6 | test | `no_adapt` | missing |  |  |  |
| chime6 | test | `teacher_ce` | missing |  |  |  |
| rev16 | dev | `no_adapt` | unavailable |  |  | Current lcasr/rev16 loader exposes test only; no dev manifest is wired into the runner. |
| rev16 | dev | `teacher_ce` | unavailable |  |  | Current lcasr/rev16 loader exposes test only; no dev manifest is wired into the runner. |
| rev16 | test | `no_adapt` | missing |  |  |  |
| rev16 | test | `teacher_ce` | missing |  |  |  |

## Regeneration

From `lcasr/`:

```bash
python results/enc_dec/rob81_floras50_finetune_eval/aggregate.py \
  --csv results/enc_dec/rob81_floras50_finetune_eval/summary.csv \
  --outcome results/enc_dec/rob81_floras50_finetune_eval/OUTCOME.md
```
