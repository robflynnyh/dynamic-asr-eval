# enc_dec_teacher_kl_relaxed_filters

Teacher-KL encoder-decoder dynamic-eval sweep with relaxed teacher filters.

This variant keeps the same teacher-KL method as `results/enc_dec/enc_dec_v2/enc_dec_teacher_kl`, but changes the teacher-quality filters to test whether the previous filter set was too aggressive.

## Method Difference

Compared with the original KL sweep:

- repeated token n-gram loop filtering checks only 6-grams instead of 2-grams and 3-grams
- the encoder-decoder/CTC agreement filter is disabled completely

The rest of the teacher/student KL update is unchanged: clean current-model teacher distribution, augmented student distribution, token-level `KL(teacher || student)`, masked over valid decoder positions.

## Sweep

| Parameter | Value |
|---|---|
| training mode | `teacher_kl` |
| teacher KL temperature | `1.0` |
| datasets | `tedlium earnings22` |
| split | `test` |
| epochs | `1` |
| repeats | `1` |
| seq / overlap | `2048 / 0` |
| learning rates | `3e-7 1e-7 1e-6` |
| augmentations | `freq6_width34_time0`, `freq3_width24_time0`, `no_aug` |
| decode | `beam=5`, `length_penalty=0.5` |
| repeated token n-gram sizes | `6` |
| CTC agreement filter | disabled |
| GPU | `2` |

## Launch

```bash
screen -L -Logfile results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_relaxed_filters/teacher_kl_relaxed_beam5_lp0p5_tedlium_earnings22.log \
  -dmS teacher_kl_relaxed_beam5_lp0p5 \
  bash -lc 'GPU=2 RESULTS_DIR=./results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_relaxed_filters TRAINING_MODE=teacher_kl TEACHER_KL_TEMPERATURE=1.0 TEACHER_REPEATED_TOKEN_NGRAM_SIZES="6" TEACHER_FILTER_CTC_AGREEMENT=0 DATASETS="tedlium earnings22" LRS="3e-7 1e-7 1e-6" AUGS="freq6_width34_time0 freq3_width24_time0 no_aug" ENC_DEC_BEAM_WIDTH=5 ENC_DEC_LENGTH_PENALTY=0.5 bash launch_scripts/tune_enc_dec_dynamic_eval_teacher_ce.sh'
```

Pickles are written as:

```text
results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_relaxed_filters/<dataset>-test-teacher_kl-beam5_lp0p5-epoch-1-lr-<lr_tag>-<aug>_<repeat>.pkl
```

Per-setting logs are written under:

```text
results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_relaxed_filters/logs/
```

## Aggregation

Compact table:

```bash
python results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_relaxed_filters/aggregate.py
```

JSON:

```bash
python results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_relaxed_filters/aggregate.py --json
```

CSV:

```bash
python results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_relaxed_filters/aggregate.py --csv results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_relaxed_filters/summary.csv
```
