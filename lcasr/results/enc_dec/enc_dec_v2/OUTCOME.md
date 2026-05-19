# enc_dec_v2 checkpoint outcome

This folder is the historical encoder-decoder checkpoint family:

```text
/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt
```

It is intentionally separate from the ROB-63 `old_seed` checkpoint:

```text
/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt
```

## Matched ROB-63 Teacher-CE Setting

The historical `enc_dec_v2/enc_dec_dynamic_eval/` artifacts include the
matched ROB-63 thesis-facing setting for TED-LIUM and Earnings22:
`teacher_ce`, `lr=1e-7`, `freq3_width24_time0`, one adaptation epoch, and
beam5/lp0.5 decode. No matching `enc_dec_v2` CHiME-6 or Rev16 rows are present
in the current result tree.

| Dataset | Split | Checkpoint | Unadapted WER | Adapted WER | Delta vs unadapted |
|---|---|---|---:|---:|---:|
| TED-LIUM | test | `enc_dec_v2` | 0.10331 | 0.10076 | -0.00255 |
| TED-LIUM | test | `old_seed` | 0.08896 | 0.08811 | -0.00085 |
| TED-LIUM | test | `rl_step_30000` | 0.08386 | 0.07992 | -0.00393 |
| Earnings22 | test | `enc_dec_v2` | 0.28724 | 0.31875 | +0.03151 |
| Earnings22 | test | `old_seed` | 0.25172 | 0.21806 | -0.03366 |
| Earnings22 | test | `rl_step_30000` | 0.23007 | 0.21365 | -0.01642 |

For this exact setting, `enc_dec_v2` improves slightly on TED-LIUM but worsens
on Earnings22. The newer `old_seed` and `rl_step_30000` checkpoint families are
therefore the more relevant ROB-63 comparison rows for this setting.

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
