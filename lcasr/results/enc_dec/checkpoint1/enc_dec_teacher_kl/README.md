# enc_dec_teacher_kl

Teacher-KL encoder-decoder dynamic-eval sweep.

This run tests whether matching the clean teacher decoder distribution is more stable than hard teacher-forced CE pseudo-label adaptation, especially on Earnings22 where CE increased insertions and substitutions.

## Method

For each dynamic-eval chunk, the runner performs an online self-distillation update:

1. Copy the chunk into augmented student inputs and one clean teacher input.
2. Decode the clean teacher input with the current encoder-decoder model using `beam=5` and `length_penalty=0.5`.
3. Treat the decoded teacher sequence as the pseudo-label prefix and apply the existing teacher-quality filters before training.
4. Run a teacher forward pass on the clean audio under `torch.no_grad()` with the decoder in eval mode, producing detached LM logits.
5. Run a student forward pass on the augmented audio copies with gradients enabled, using the same teacher-forced token prefix.
6. Minimize token-level `KL(teacher || student)`:

   ```text
   teacher_probs = softmax(teacher_logits / temperature)
   student_log_probs = log_softmax(student_logits / temperature)
   loss = KL(teacher_probs || student_log_probs)
   ```

   Padding positions are masked, the loss is averaged over valid decoder positions and augmented copies, and the result is multiplied by `temperature ** 2`.
7. Apply one optimizer step to the current model.

This differs from `teacher_ce`: CE collapses the target to one hard pseudo-label token per position, while KL preserves the teacher distribution over the full vocabulary. The teacher is not a separately frozen checkpoint; it is the current adapted model state immediately before the chunk update.

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
| GPU | `1` |

## Launch

```bash
screen -L -Logfile results/enc_dec/checkpoint1/enc_dec_teacher_kl/teacher_kl_beam5_lp0p5_tedlium_earnings22.log \
  -dmS teacher_kl_beam5_lp0p5 \
  bash -lc 'GPU=1 RESULTS_DIR=./results/enc_dec/checkpoint1/enc_dec_teacher_kl TRAINING_MODE=teacher_kl TEACHER_KL_TEMPERATURE=1.0 DATASETS="tedlium earnings22" LRS="3e-7 1e-7 1e-6" AUGS="freq6_width34_time0 freq3_width24_time0 no_aug" ENC_DEC_BEAM_WIDTH=5 ENC_DEC_LENGTH_PENALTY=0.5 bash launch_scripts/tune_enc_dec_dynamic_eval_teacher_ce.sh'
```

Pickles are written as:

```text
results/enc_dec/checkpoint1/enc_dec_teacher_kl/<dataset>-test-teacher_kl-beam5_lp0p5-epoch-1-lr-<lr_tag>-<aug>_<repeat>.pkl
```

Per-setting logs are written under:

```text
results/enc_dec/checkpoint1/enc_dec_teacher_kl/logs/
```

## Aggregation

Compact table:

```bash
python results/enc_dec/checkpoint1/enc_dec_teacher_kl/aggregate.py
```

JSON:

```bash
python results/enc_dec/checkpoint1/enc_dec_teacher_kl/aggregate.py --json
```

CSV:

```bash
python results/enc_dec/checkpoint1/enc_dec_teacher_kl/aggregate.py --csv results/enc_dec/checkpoint1/enc_dec_teacher_kl/summary.csv
```
