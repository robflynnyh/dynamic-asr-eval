# TEDLIUM Dev Epoch-Relabel KL Stability Sweep

Goal: find a teacher-KL epoch-relabel setting whose improvement is less
repeat-sensitive than the current quick-test setup.

## Fixed Setup

| Axis | Value |
|---|---|
| dataset / split | `tedlium / dev` |
| model | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt` |
| training mode | `teacher_kl` |
| teacher/student timing | `--teacher_epoch_relabel` |
| epochs | `1` |
| repeats | `1` by default |
| chunking | `seq=2048`, `overlap=0` |
| decode | `beam=5`, `enc_dec_length_penalty=0.5` |
| output folder | `results/enc_dec/enc_dec_teacher_epoch_relabel` |

This intentionally uses TEDLIUM dev before test so the sweep is a tuning pass,
not another test-set search.

## Primary Grid

Learning rate:

| Label | Value | Reason |
|---|---:|---|
| high | `3e-7` | previous broad-grid LR; useful bridge between aggressive and current |
| current | `1e-7` | best-looking KL setting so far |
| very gentle | `1e-8` | check whether very small updates give more stable but weaker adaptation |

KL temperature:

| Value | Interpretation |
|---:|---|
| `1.0` | current soft teacher distribution |
| `0.7` | sharper but still soft |
| `0.5` | closer to hard pseudo-labeling |

Filtering:

| Label | Filters |
|---|---|
| `relaxed` | max length, max consecutive token repeat, repeated token 6-grams, repeated words |
| `ctc` | `relaxed` plus CTC agreement |
| `strict_ctc` | CTC agreement plus repeated token 2/3-gram loop filtering |

SpecAugment:

| Label | Frequency masks | Time masks | Purpose |
|---|---:|---:|---|
| `no_aug` | `0 x width34` | `0` | no augmentation control |
| `freq2_width16_time0` | `2 x width16` | `0` | weaker frequency masking |
| `freq3_width24_time0` | `3 x width24` | `0` | current quick-test setting |
| `freq6_width34_time0` | `6 x width34` | `0` | previously strong setting that looked promising |
| `freq8_width48_time0` | `8 x width48` | `0` | extra-strong frequency-only stress test |
| `freq2_width16_time1` | `2 x width16` | `1` | weak frequency masking plus time masking |

Primary grid size: `3 LRs x 3 KL temps x 3 filters x 6 augmentations = 162`
settings. With the default `REPEATS=1`, this produces 162 TEDLIUM-dev
adaptation/eval runs plus one no-adapt baseline. After this broad pass, rerun
`REPEATS=3` or higher for the best few settings.

## Optional Additions

Only add these if the primary grid does not settle the stability question:

| Axis | Value | Reason |
|---|---|---|
| filtering | `lowconf_ctc` | adds forced-path confidence filtering |
| augmentation | `freq3_width24_time1` | current frequency masking plus one time mask |
| augmentation | `freq6_width34_time1` | strong frequency masking plus one time mask |
| augmentation | `time1_only` | isolates the effect of time masking |

## Launcher

Default primary sweep:

```bash
GPU=0 bash launch_scripts/tune_enc_dec_teacher_epoch_relabel_tedlium_dev.sh
```

Dry run:

```bash
DRY_RUN=1 GPU=0 bash launch_scripts/tune_enc_dec_teacher_epoch_relabel_tedlium_dev.sh
```

Include the optional low-confidence and time-mask additions:

```bash
GPU=0 \
FILTERS="relaxed ctc strict_ctc lowconf_ctc" \
AUGS="no_aug freq2_width16_time0 freq3_width24_time0 freq6_width34_time0 freq8_width48_time0 freq2_width16_time1 freq3_width24_time1 freq6_width34_time1 time1_only" \
bash launch_scripts/tune_enc_dec_teacher_epoch_relabel_tedlium_dev.sh
```

The `time1` suffix means `spec_augment_n_time_masks=1`. Since this runner does
not set `spec_augment_time_mask_param`, `lib.py` uses its default
`spec_augment_min_p=0.05`, so the single time mask has a maximum width of about
5% of the chunk time axis. The mask value is the spectrogram mean unless
`spec_augment_zero_masking=True` is explicitly supplied.

For a smaller smoke sweep before the full grid:

```bash
GPU=0 \
REPEATS=1 \
LRS="1e-7 5e-8" \
KL_TEMPS="1.0 0.7" \
FILTERS="relaxed ctc" \
AUGS="freq2_width16_time0 freq2_width16_time1" \
bash launch_scripts/tune_enc_dec_teacher_epoch_relabel_tedlium_dev.sh
```

## Selection Rule

Pick by mean TEDLIUM-dev WER, but only trust settings with low repeat variance.
The target is not just the largest single improvement; it is a setting with:

- negative mean delta versus beam-search no-adapt baseline,
- lower `wer_std` across repeats than the current quick-test setup,
- no obvious insertion/deletion imbalance,
- enough retained teacher labels in the logs to explain the update.
