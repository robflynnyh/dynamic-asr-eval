# ROB-63 Earnings22 unadapted sanity check

This reruns unadapted Earnings22 `test` beam5/lp0.5 decoding for the ROB-63 old seed and the older checkpoint used by the historical `enc_dec/OUTCOME` baseline.

## Checkpoint Key

- `old_seed`: `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`
- `enc_dec_outcome_seed`: `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt`

## Result

| Dataset | Split | Decode | Checkpoint | WER | Delta vs old seed | Relative delta | Ins | Del | Sub |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| earnings22 | test | beam5_lp0p5 | enc_dec_outcome_seed | 0.28724 | +0.03552 | +14.11% | 0.04789 | 0.05935 | 0.17999 |
| earnings22 | test | beam5_lp0p5 | old_seed | 0.25172 | +0.00000 | +0.00% | 0.04352 | 0.05518 | 0.15301 |

## Interpretation

The rerun gap is +0.03552 absolute WER, with the older `enc_dec_outcome_seed` checkpoint at 0.28724 WER and the ROB-63 `old_seed` checkpoint at 0.25172 WER.
