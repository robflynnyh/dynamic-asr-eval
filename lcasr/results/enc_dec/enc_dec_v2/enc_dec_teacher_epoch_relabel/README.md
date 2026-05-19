# enc_dec_teacher_epoch_relabel

Encoder-decoder dynamic-eval teacher relabel ablation.

This folder is for the opt-in `--teacher_epoch_relabel` path. It keeps the
existing teacher-filtered CE/KL setup, but changes teacher/student timing:

1. At the start of each adaptation epoch, the current model is used as a fixed
   teacher.
2. The teacher labels every chunk before any student update in that epoch.
3. Existing teacher-quality filters remove bad examples.
4. The student trains for one epoch on the retained examples.
5. The next epoch repeats the label/filter/train cycle with the updated model
   as the new teacher.

This is intended to test whether separating teacher labeling from student
updates is more stable than the default chunk-by-chunk setup where the teacher
changes immediately after each update.

## Launch

For the TEDLIUM-dev stability grid, see
[`SWEEP_PLAN.md`](SWEEP_PLAN.md). The dedicated launcher is:

```bash
GPU=0 bash launch_scripts/tune_enc_dec_teacher_epoch_relabel_tedlium_dev.sh
```

For the older broad dataset/LR/augmentation grid:

```bash
GPU=1 \
RESULTS_DIR=./results/enc_dec/enc_dec_v2/enc_dec_teacher_epoch_relabel \
TRAINING_MODE=teacher_kl \
TEACHER_EPOCH_RELABEL=1 \
DATASETS="tedlium earnings22" \
LRS="3e-7 1e-7 1e-6" \
AUGS="freq6_width34_time0 freq3_width24_time0 no_aug" \
ENC_DEC_BEAM_WIDTH=5 \
ENC_DEC_LENGTH_PENALTY=0.5 \
bash launch_scripts/tune_enc_dec_dynamic_eval_teacher_ce.sh
```

Use `TRAINING_MODE=teacher_ce` for hard teacher-forced CE with the same
epoch-relabel timing.

## Files

```text
<dataset>-test-<teacher_ce_epoch_relabel|teacher_kl_epoch_relabel>-beam5_lp0p5-epoch-<E>-lr-<lr_tag>-<aug>_<repeat>.pkl
```

Per-setting logs are written under:

```text
results/enc_dec/enc_dec_v2/enc_dec_teacher_epoch_relabel/logs/
```

## Aggregation

```bash
python results/enc_dec/enc_dec_v2/enc_dec_teacher_epoch_relabel/aggregate.py
```
