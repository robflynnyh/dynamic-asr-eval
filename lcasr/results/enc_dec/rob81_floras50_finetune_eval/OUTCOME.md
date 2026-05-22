# ROB-122 ROB-81 Floras-finetuned encoder-decoder eval

Fixed adaptation setting: `teacher_ce`, `lr=1e-7`, `freq3_width24_time0`, `beam5_lp0p5`, epoch `1`, no filtering.
Matched baseline setting: `no_adapt`, `beam5_lp0p5`, epoch `0`, `optim_lr=0.0`, no augmentation.

Completed rows: 14/16. Missing rows: 0. Unavailable rows: 2.

Rev16 dev rows are marked unavailable because the current `lcasr/rev16` loader exposes test only.

## Checkpoint

`/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0/step_323484.pt`

## Rows

| Dataset | Split | Mode | Status | WER | Delta vs no-adapt | Source / note |
|---|---|---|---|---:|---:|---|
| tedlium | dev | `no_adapt` | complete | 0.08539 |  | results/enc_dec/rob81_floras50_finetune_eval/pkl/tedlium-dev-rob81_floras50-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug_1.pkl |
| tedlium | dev | `teacher_ce` | complete | 0.07688 | -0.00851 | results/enc_dec/rob81_floras50_finetune_eval/pkl/tedlium-dev-rob81_floras50-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| tedlium | test | `no_adapt` | complete | 0.07613 |  | results/enc_dec/rob81_floras50_finetune_eval/pkl/tedlium-test-rob81_floras50-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug_1.pkl |
| tedlium | test | `teacher_ce` | complete | 0.07113 | -0.00500 | results/enc_dec/rob81_floras50_finetune_eval/pkl/tedlium-test-rob81_floras50-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| earnings22 | dev | `no_adapt` | complete | 0.28863 |  | results/enc_dec/rob81_floras50_finetune_eval/pkl/earnings22-dev-rob81_floras50-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug_1.pkl |
| earnings22 | dev | `teacher_ce` | complete | 0.44678 | 0.15815 | results/enc_dec/rob81_floras50_finetune_eval/pkl/earnings22-dev-rob81_floras50-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| earnings22 | test | `no_adapt` | complete | 0.20828 |  | results/enc_dec/rob81_floras50_finetune_eval/pkl/earnings22-test-rob81_floras50-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug_1.pkl |
| earnings22 | test | `teacher_ce` | complete | 0.18091 | -0.02737 | results/enc_dec/rob81_floras50_finetune_eval/pkl/earnings22-test-rob81_floras50-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| chime6 | dev | `no_adapt` | complete | 0.81509 |  | results/enc_dec/rob81_floras50_finetune_eval/pkl/chime6-dev-rob81_floras50-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug_1.pkl |
| chime6 | dev | `teacher_ce` | complete | 1.00000 | 0.18491 | results/enc_dec/rob81_floras50_finetune_eval/pkl/chime6-dev-rob81_floras50-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| chime6 | test | `no_adapt` | complete | 0.84790 |  | results/enc_dec/rob81_floras50_finetune_eval/pkl/chime6-test-rob81_floras50-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug_1.pkl |
| chime6 | test | `teacher_ce` | complete | 1.00000 | 0.15210 | results/enc_dec/rob81_floras50_finetune_eval/pkl/chime6-test-rob81_floras50-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| rev16 | dev | `no_adapt` | unavailable |  |  | Current lcasr/rev16 loader exposes test only; no dev manifest is wired into the runner. |
| rev16 | dev | `teacher_ce` | unavailable |  |  | Current lcasr/rev16 loader exposes test only; no dev manifest is wired into the runner. |
| rev16 | test | `no_adapt` | complete | 0.17839 |  | results/enc_dec/rob81_floras50_finetune_eval/pkl/rev16-test-rob81_floras50-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug_1.pkl |
| rev16 | test | `teacher_ce` | complete | 0.68917 | 0.51078 | results/enc_dec/rob81_floras50_finetune_eval/pkl/rev16-test-rob81_floras50-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |

## Regeneration

From `lcasr/`:

```bash
python results/enc_dec/rob81_floras50_finetune_eval/aggregate.py \
  --csv results/enc_dec/rob81_floras50_finetune_eval/summary.csv \
  --outcome results/enc_dec/rob81_floras50_finetune_eval/OUTCOME.md
```
