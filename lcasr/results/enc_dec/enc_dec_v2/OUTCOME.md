# enc_dec_v2 checkpoint outcome

This folder is the historical encoder-decoder checkpoint family:

```text
/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt
```

It is intentionally separate from the ROB-63 `old_seed` checkpoint:

```text
/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt
```

## Earnings22 Sanity Check

The dedicated ROB-63 sanity rerun uses the same unadapted beam5/lp0.5 decode
for `enc_dec_v2` and `old_seed` on Earnings22 `test`:

| Checkpoint key | WER |
|---|---:|
| `enc_dec_outcome_seed` (`enc_dec_v2/step_105360.pt`) | 0.28724 |
| `old_seed` (`enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`) | 0.25172 |

The 28% versus 25% Earnings22 gap is therefore a checkpoint-family difference,
not a reporting path mix-up.

## Historical Results

Use the child folder outcomes for the older checkpoint-family ablations:

| Folder | Main output |
|---|---|
| `enc_dec_beam_tedlium_dev/` | `enc_dec_beam_tedlium_dev/OUTCOME.md` |
| `enc_dec_dynamic_eval/` | `enc_dec_dynamic_eval/OUTCOME.md` |
| `enc_dec_teacher_kl/` | `enc_dec_teacher_kl/OUTCOME.md` |
| `enc_dec_teacher_kl_entropy_filter/` | `enc_dec_teacher_kl_entropy_filter/OUTCOME.md` |
| `enc_dec_teacher_kl_relaxed_filters/` | `enc_dec_teacher_kl_relaxed_filters/OUTCOME.md` |
| `rob63_earnings_unadapted_sanity/` | `rob63_earnings_unadapted_sanity/OUTCOME.md` |

For the current ROB-63 seed-vs-RL comparison, use `../old_seed/OUTCOME.md`,
`../rl_step_30000/OUTCOME.md`, and
`../rl_step_30000/rob63_rl_self_training_compare/COMBINED_OUTCOME.md`.
