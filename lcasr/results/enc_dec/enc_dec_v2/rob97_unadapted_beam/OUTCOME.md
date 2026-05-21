# ROB-97 enc_dec_v2 unadapted beam-search rows

Fixed setting: `no_adapt`, `beam5_lp0p5`, epoch `0`, no augmentation.

Completed rows: 2/2. Missing rows: 0.

## ROB-96 Rows

| Checkpoint | Dataset | Split | Status | WER | Source / note |
|---|---|---|---|---:|---|
| enc_dec_v2 | chime6 | test | complete | 0.86331 | results/enc_dec/enc_dec_v2/rob97_unadapted_beam/pkl/chime6-test-enc_dec_v2-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug_1.pkl |
| enc_dec_v2 | rev16 | test | complete | 0.19109 | results/enc_dec/enc_dec_v2/rob97_unadapted_beam/pkl/rev16-test-enc_dec_v2-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug_1.pkl |

## Checkpoint

- `enc_dec_v2`: `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt`

## Regeneration

From `lcasr/`:

```bash
python results/enc_dec/enc_dec_v2/rob97_unadapted_beam/aggregate.py \
  --csv results/enc_dec/enc_dec_v2/rob97_unadapted_beam/summary.csv \
  --outcome results/enc_dec/enc_dec_v2/rob97_unadapted_beam/OUTCOME.md
```
