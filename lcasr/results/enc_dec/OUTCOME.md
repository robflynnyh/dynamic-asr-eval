# Encoder-decoder results outcome

This file summarizes the current child `OUTCOME.md` files under
`results/enc_dec/`. All reported rows are single-run snapshots unless a child
folder states otherwise.

## Comparability Rule

Do not compare WERs across greedy and beam-search baselines as though they are
the same baseline. A same-decode no-adapt/default row is required for a clean
adaptation delta.

The current adaptation sweeps generally use `beam=5`, `length_penalty=0.5`.
The decode sweep folder mixes greedy/default and beam settings by design.

Checkpoint provenance also matters. The older `enc_dec_dynamic_eval` result
folder uses `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt`.
ROB-63-specific outcome files use these explicit checkpoint labels:

- `old_seed`: normal encoder-decoder seed checkpoint at `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt`.
- `rl_step_30000`: 30K RL-trained checkpoint at `/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt`.

Use the per-folder checkpoint key and `checkpoint_path` columns in ROB-63
summary CSVs before comparing rows across outcome files.

## Decode Sweep

Folder: `enc_dec_beam_tedlium_dev`

Best current saved decode on TEDLIUM dev:

```text
beam10_lp0p5: 11.15% WER
```

Recommended quality/runtime decode:

```text
beam5_lp0p5: 11.20% WER
```

`beam10_lp0p5` is slightly better by WER, but the 0.05 absolute WER gain is
small. `beam5_lp0p5` remains the practical default used by the adaptation
sweeps. The clearly bad settings are `max_generate=80` and the stronger
no-repeat/repetition-penalty variants.

## Beam No-Adapt Baselines

Folder: `enc_dec_dynamic_eval`

Available encoder-decoder no-adapt results on test with beam search:

| Dataset | Best WER | Best setting | Interpretation |
|---|---:|---|---|
| `tedlium` | 10.33 | `tedlium-test-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug` | Test-set no-adapt baseline with `beam=5`, `length_penalty=0.5`; error split is 1.44 ins / 3.62 del / 5.28 sub. |
| `earnings22` | 28.72 | `earnings22-test-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug` | Test-set no-adapt baseline with `beam=5`, `length_penalty=0.5`; error split is 4.79 ins / 5.94 del / 18.00 sub. |

These are the same-decode baselines used for the teacher-CE adaptation
comparisons below.

## Teacher-CE Adaptation

Folder: `enc_dec_dynamic_eval`

This folder has matching beam5/lp0.5 no-adapt baselines:

| Dataset | No-adapt WER | Best adapted WER | Best setting | Interpretation |
|---|---:|---:|---|---|
| `tedlium` | 10.33 | 10.08 | `lr=1e-7`, `freq3_width24_time0` | Small improvement. |
| `earnings22` | 28.72 | 31.44 | `lr=3e-7`, `freq3_width24_time0` | Still worse than baseline. |

Teacher-forced CE helps TEDLIUM modestly but is unsafe on Earnings22. The
Earnings22 `no_aug` rows are especially bad, with large insertion and
substitution growth.

## Teacher-KL

Folder: `enc_dec_teacher_kl`

This folder does not include local same-decode no-adapt baselines, so values
below are absolute WERs.

| Dataset | Best WER | Best setting | Interpretation |
|---|---:|---|---|
| `tedlium` | 10.24 | `lr=1e-7`, `freq6_width34_time0` | Stable, small gains at best. |
| `earnings22` | 28.73 | `lr=1e-6`, `no_aug` | Much safer than hard CE; mostly near no-adapt behavior. |

The `1e-5` TEDLIUM follow-up rows are clear failures due to high deletion.

## Teacher-KL With Entropy Filter

Folder: `enc_dec_teacher_kl_entropy_filter`

| Dataset | Best WER | Best setting | Interpretation |
|---|---:|---|---|
| `tedlium` | 10.28 | `lr=1e-7`, `freq6_width34_time0` | Stable, but not better than original KL. |
| `earnings22` | 28.74 | `no_aug` rows | Similar to original KL; augmented rows still raise WER. |

The entropy/low-confidence filter is not harmful, but it does not clearly beat
the original teacher-KL sweep.

## Teacher-KL With Relaxed Filters

Folder: `enc_dec_teacher_kl_relaxed_filters`

| Dataset | Best WER | Best setting | Interpretation |
|---|---:|---|---|
| `tedlium` | 10.29 | `lr=3e-7`, `freq6_width34_time0` | Not better than original KL on TEDLIUM. |
| `earnings22` | 28.06 | `lr=1e-7`, `freq3_width24_time0` | Best KL result for Earnings22 among these folders. |

The relaxed-filter variant is the most promising KL variant for Earnings22,
but it should be compared against a same-decode no-adapt baseline in the same
folder before treating the improvement as final.

## Overall Takeaway

Use `beam=5`, `length_penalty=0.5` as the practical default decode setting.
Hard teacher-CE is useful only on the easier TEDLIUM setting and fails on
Earnings22. Teacher-KL is safer on Earnings22, and the relaxed-filter KL
variant is the strongest saved Earnings22 result in this grouped set.
