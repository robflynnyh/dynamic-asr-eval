# enc_dec_teacher_kl_entropy_filter

Teacher-KL encoder-decoder dynamic-eval sweep with teacher low-confidence filtering enabled.

This variant keeps the original KL sweep filter set and additionally skips teacher updates when the teacher-forced decoder distribution is too uncertain.

## Filter Difference

Compared with `results/enc_dec/enc_dec_v2/enc_dec_teacher_kl`, this run enables:

```text
--teacher_filter_low_confidence
--teacher_min_mean_max_prob 0.35
--teacher_max_mean_entropy 2.5
```

The existing filters remain enabled:

```text
--teacher_filter_max_length
--teacher_filter_max_consecutive_token_repeat
--teacher_filter_repeated_token_ngrams
--teacher_repeated_token_ngram_sizes 2 3
--teacher_filter_repeated_words
--teacher_filter_ctc_agreement
```

The teacher/student KL update is unchanged: clean current-model teacher distribution, augmented student distribution, token-level `KL(teacher || student)`, masked over valid decoder positions.

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
| mean max prob threshold | `0.35` |
| mean entropy threshold | `2.5` |
| GPU | `0` |

## Launch

```bash
screen -L -Logfile results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_entropy_filter/teacher_kl_entropy_beam5_lp0p5_tedlium_earnings22.log \
  -dmS teacher_kl_entropy_beam5_lp0p5 \
  bash -lc 'GPU=0 RESULTS_DIR=./results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_entropy_filter TRAINING_MODE=teacher_kl TEACHER_KL_TEMPERATURE=1.0 TEACHER_FILTER_LOW_CONFIDENCE=1 TEACHER_MIN_MEAN_MAX_PROB=0.35 TEACHER_MAX_MEAN_ENTROPY=2.5 DATASETS="tedlium earnings22" LRS="3e-7 1e-7 1e-6" AUGS="freq6_width34_time0 freq3_width24_time0 no_aug" ENC_DEC_BEAM_WIDTH=5 ENC_DEC_LENGTH_PENALTY=0.5 bash launch_scripts/tune_enc_dec_dynamic_eval_teacher_ce.sh'
```

Pickles are written as:

```text
results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_entropy_filter/<dataset>-test-teacher_kl-beam5_lp0p5-epoch-1-lr-<lr_tag>-<aug>_<repeat>.pkl
```

Per-setting logs are written under:

```text
results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_entropy_filter/logs/
```

## Aggregation

Compact table:

```bash
python results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_entropy_filter/aggregate.py
```

JSON:

```bash
python results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_entropy_filter/aggregate.py --json
```

CSV:

```bash
python results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_entropy_filter/aggregate.py --csv results/enc_dec/enc_dec_v2/enc_dec_teacher_kl_entropy_filter/summary.csv
```
