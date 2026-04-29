# enc_dec_dynamic_eval

Encoder-decoder test-time adaptation experiments comparing teacher-forced
adaptation and CTC-style auxiliary adaptation under noisy pseudo-labels.

- Adaptive launcher: `launch_scripts/tune_enc_dec_dynamic_eval_adaptive.sh`
- Teacher-CE ablation launcher:
  `launch_scripts/tune_enc_dec_dynamic_eval_teacher_ce.sh`
- Runner: `enc_dec_dynamic_eval_test.py`
- Checkpoint default:
  `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt`

## Question

Can encoder-decoder dynamic evaluation recover the noise robustness seen in
the CTC pseudo-labeling setup?

Early single-recording observations:

| Dataset | Baseline | `teacher_ce` | `ctc_aux` |
|---|---:|---:|---:|
| TEDLIUM-like easier recording | 11.08 | 9.6 | 11.2 |
| Earnings22-like noisier recording | 29.7 | 33.3 | 28.3 |

This suggests teacher-forced CE is useful when the pseudo-label is reliable,
but unsafe on noisy recordings. CTC auxiliary adaptation is less helpful on
clean/easy data but more robust on hard data.

## Adaptive Mode

The main sweep uses:

```bash
--training_mode adaptive_ce_ctc_aux
```

For each chunk:

1. Decode the clean chunk once to get the teacher pseudo-label.
2. Decode the same clean chunk again with sampling at
   `--teacher_decode_agreement_temperature`.
3. Compute `1 - CER` agreement between the sampled decode and the teacher
   decode.
4. If agreement is at least `--teacher_decode_agreement_min_similarity`, train
   with `teacher_ce`.
5. Otherwise train with `ctc_aux`.

The default launcher threshold is strict:

```bash
--teacher_decode_agreement_min_similarity 0.90
--teacher_decode_agreement_temperature 0.7
```

Do not add `--teacher_filter_decode_agreement` for this adaptive run unless
you intentionally want low-agreement chunks to be skipped instead of routed to
`ctc_aux`.

## Safety Filters

The launcher keeps the hard teacher-quality filters from the current manual
command:

```bash
--teacher_filter_max_length
--teacher_filter_max_consecutive_token_repeat
--teacher_filter_repeated_token_ngrams
--teacher_filter_repeated_words
--teacher_filter_ctc_agreement
```

These filters still skip pathological teacher outputs before the adaptive
CE/CTC routing decision.

## Sweep

Defaults in `launch_scripts/tune_enc_dec_dynamic_eval_adaptive.sh`:

| Parameter | Values |
|---|---|
| datasets | `tedlium earnings22` |
| split | `test` |
| epochs | `1` |
| repeats | `1` |
| seq / overlap | `2048 / 0` |
| learning rates | `3e-7 1e-7 1e-6` |
| augmentations | `freq6_width34_time0`, `freq3_width24_time0`, `no_aug` |

Augmentation definitions:

| Tag | `spec_augment_n_freq_masks` | `spec_augment_freq_mask_param` | `spec_augment_n_time_masks` |
|---|---:|---:|---:|
| `freq6_width34_time0` | 6 | 34 | 0 |
| `freq3_width24_time0` | 3 | 24 | 0 |
| `no_aug` | 0 | 34 | 0 |

The first row is the current manual setting.

## Running

Default adaptive sweep:

```bash
bash launch_scripts/tune_enc_dec_dynamic_eval_adaptive.sh
```

Override GPU, datasets, LRs, or augmentations:

```bash
GPU=1 \
DATASETS="tedlium earnings22" \
LRS="3e-7 1e-7 1e-6" \
AUGS="freq6_width34_time0 freq3_width24_time0 no_aug" \
bash launch_scripts/tune_enc_dec_dynamic_eval_adaptive.sh
```

Print the commands without running them:

```bash
DRY_RUN=1 bash launch_scripts/tune_enc_dec_dynamic_eval_adaptive.sh
DRY_RUN=1 bash launch_scripts/tune_enc_dec_dynamic_eval_teacher_ce.sh
```

Teacher-forced CE ablation sweep:

```bash
bash launch_scripts/tune_enc_dec_dynamic_eval_teacher_ce.sh
```

This uses the same datasets, learning-rate grid, augmentation grid, checkpoint,
and hard teacher filters as the adaptive sweep, but always trains with:

```bash
--training_mode teacher_ce
```

The ablation is intended to quantify where plain teacher-forced pseudo-label
adaptation helps, and where it fails under noisier pseudo-labels.

## Files

Pickles are written as:

```text
results/enc_dec_dynamic_eval/<dataset>-<split>-adaptive_ce_ctc_aux-epoch-<E>-lr-<lr_tag>-<aug>-agree<threshold>.pkl
```

Teacher-CE ablation pickles are written as:

```text
results/enc_dec_dynamic_eval/<dataset>-<split>-teacher_ce-epoch-<E>-lr-<lr_tag>-<aug>.pkl
```

With repeats, `enc_dec_dynamic_eval_test.py` appends `_<repeat>` before
`.pkl`.

Logs are written under:

```text
results/enc_dec_dynamic_eval/logs/
```

## Aggregation

Compact table:

```bash
python results/enc_dec_dynamic_eval/aggregate.py
```

JSON:

```bash
python results/enc_dec_dynamic_eval/aggregate.py --json
```

CSV:

```bash
python results/enc_dec_dynamic_eval/aggregate.py --csv results/enc_dec_dynamic_eval/summary.csv
```
