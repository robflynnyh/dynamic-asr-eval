# ROB-94 fixed-setting encoder-decoder thesis rows

Fixed setting: `teacher_ce`, `lr=1e-7`, `freq3_width24_time0`, `beam5_lp0p5`, epoch `1`, no filtering.

Completed rows: 21/24. Missing rows: 0. Unavailable rows: 3.

Rev16 dev rows are marked unavailable because the current `lcasr/rev16` loader and `enc_dec_dynamic_eval_test.py` expose Rev16 test only.

## Thesis Table

| Checkpoint | Dataset | Split | Status | WER | Source / note |
|---|---|---|---|---:|---|
| enc_dec_v2 | tedlium | dev | complete | 0.11402 | results/enc_dec/rob94_fixed_setting_thesis/pkl/tedlium-dev-enc_dec_v2-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| enc_dec_v2 | tedlium | test | complete | 0.10076 | results/enc_dec/enc_dec_v2/enc_dec_dynamic_eval/tedlium-test-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| enc_dec_v2 | earnings22 | dev | complete | 0.97129 | results/enc_dec/rob94_fixed_setting_thesis/pkl/earnings22-dev-enc_dec_v2-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| enc_dec_v2 | earnings22 | test | complete | 0.31875 | results/enc_dec/enc_dec_v2/enc_dec_dynamic_eval/earnings22-test-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| enc_dec_v2 | chime6 | dev | complete | 1.00000 | results/enc_dec/rob94_fixed_setting_thesis/pkl/chime6-dev-enc_dec_v2-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| enc_dec_v2 | chime6 | test | complete | 1.00000 | results/enc_dec/rob94_fixed_setting_thesis/pkl/chime6-test-enc_dec_v2-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| enc_dec_v2 | rev16 | dev | unavailable |  | Current lcasr/rev16 loader exposes test only; no dev manifest is wired into the runner. |
| enc_dec_v2 | rev16 | test | complete | 1.13088 | results/enc_dec/rob94_fixed_setting_thesis/pkl/rev16-test-enc_dec_v2-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| old_seed | tedlium | dev | complete | 0.10755 | results/enc_dec/rob94_fixed_setting_thesis/pkl/tedlium-dev-old_seed-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| old_seed | tedlium | test | complete | 0.08811 | results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/pkl/tedlium-test-old_seed-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| old_seed | earnings22 | dev | complete | 0.58824 | results/enc_dec/rob94_fixed_setting_thesis/pkl/earnings22-dev-old_seed-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| old_seed | earnings22 | test | complete | 0.21806 | results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/pkl/earnings22-test-old_seed-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| old_seed | chime6 | dev | complete | 1.00000 | results/enc_dec/rob94_fixed_setting_thesis/pkl/chime6-dev-old_seed-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| old_seed | chime6 | test | complete | 1.00000 | results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/pkl/chime6-test-old_seed-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| old_seed | rev16 | dev | unavailable |  | Current lcasr/rev16 loader exposes test only; no dev manifest is wired into the runner. |
| old_seed | rev16 | test | complete | 0.23790 | results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/pkl/rev16-test-old_seed-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| rl_step_30000 | tedlium | dev | complete | 0.09274 | results/enc_dec/rob94_fixed_setting_thesis/pkl/tedlium-dev-rl_step_30000-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| rl_step_30000 | tedlium | test | complete | 0.07992 | results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/pkl/tedlium-test-rl_step_30000-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| rl_step_30000 | earnings22 | dev | complete | 0.86798 | results/enc_dec/rob94_fixed_setting_thesis/pkl/earnings22-dev-rl_step_30000-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| rl_step_30000 | earnings22 | test | complete | 0.21365 | results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/pkl/earnings22-test-rl_step_30000-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| rl_step_30000 | chime6 | dev | complete | 1.00000 | results/enc_dec/rob94_fixed_setting_thesis/pkl/chime6-dev-rl_step_30000-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| rl_step_30000 | chime6 | test | complete | 1.00000 | results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/pkl/chime6-test-rl_step_30000-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |
| rl_step_30000 | rev16 | dev | unavailable |  | Current lcasr/rev16 loader exposes test only; no dev manifest is wired into the runner. |
| rl_step_30000 | rev16 | test | complete | 0.23599 | results/enc_dec/rl_step_30000/rob63_best_ce_remaining_datasets/pkl/rev16-test-rl_step_30000-teacher_ce-beam5_lp0p5-epoch-1-lr-1em7-freq3_width24_time0_1.pkl |

## Checkpoints

- `enc_dec_v2`: `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt`
- `old_seed`: `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`
- `rl_step_30000`: `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt`

## Regeneration

From `lcasr/`:

```bash
python results/enc_dec/rob94_fixed_setting_thesis/aggregate.py \
  --csv results/enc_dec/rob94_fixed_setting_thesis/summary.csv \
  --outcome results/enc_dec/rob94_fixed_setting_thesis/OUTCOME.md
```
