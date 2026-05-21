# ROB-63 Earnings22 unadapted sanity check

This folder reruns the Earnings22 `test` unadapted beam5/lp0.5 baseline for two encoder-decoder checkpoints:

- `old_seed`: `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`
- `enc_dec_outcome_seed`: `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt`

The run checks whether the previously reported Earnings22 gap, roughly 25% WER for the ROB-63 old seed versus roughly 28% WER for the older `enc_dec/OUTCOME` baseline, is a real checkpoint difference under the same unadapted decode.

Launch command:

```bash
/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob63_earnings_unadapted_sanity_queued.sh
```

Expected committed outputs after completion:

- `pkl/*.pkl`
- `summary.csv`
- `OUTCOME.md`

Runtime logs are intentionally not committed.

Completed result:

- `enc_dec_outcome_seed`: `0.28724` WER
- `old_seed`: `0.25172` WER

This confirms the historical 28% versus ROB-63 25% Earnings22 gap is a
checkpoint-family difference under the same beam5/lp0.5 unadapted decode.
