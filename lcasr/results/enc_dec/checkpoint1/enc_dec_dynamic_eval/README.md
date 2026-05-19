# enc_dec_dynamic_eval

Encoder-decoder test-time adaptation experiments comparing teacher-forced
adaptation and CTC-style auxiliary adaptation under noisy pseudo-labels.

Current fresh run target: teacher-forced CE only, using the TEDLIUM-dev
beam-search decode setting (`beam=5`, `length_penalty=0.5`) for both teacher
pseudo-label decoding and final transcript decoding.

- Adaptive launcher: `launch_scripts/tune_enc_dec_dynamic_eval_adaptive.sh`
- Teacher-CE / teacher-KL ablation launcher:
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

Default grid used by the adaptive and teacher-CE launchers:

| Parameter | Values |
|---|---|
| datasets | `tedlium earnings22` |
| split | `test` |
| epochs | `1` |
| repeats | `1` |
| seq / overlap | `2048 / 0` |
| learning rates | `3e-7 1e-7 1e-6` |
| augmentations | `freq6_width34_time0`, `freq3_width24_time0`, `no_aug` |
| encoder-decoder decode | greedy unless overridden |

Augmentation definitions:

| Tag | `spec_augment_n_freq_masks` | `spec_augment_freq_mask_param` | `spec_augment_n_time_masks` |
|---|---:|---:|---:|
| `freq6_width34_time0` | 6 | 34 | 0 |
| `freq3_width24_time0` | 3 | 24 | 0 |
| `no_aug` | 0 | 34 | 0 |

The first row is the current manual setting.

## Encoder-Decoder Beam Decode

The dynamic-eval runner accepts the same autoregressive encoder-decoder beam
search flags used by the TEDLIUM dev decode sweep:

```bash
--enc_dec_beam_width 5
--enc_dec_length_penalty 0.5
```

When these flags are set, they are used for both:

- the deterministic teacher pseudo-label decoded before each adaptation update
- the final decoded transcript after adaptation

The sampled agreement decode used by `adaptive_ce_ctc_aux` remains sampled, so
the adaptive CE/CTC routing still compares the beam teacher against an
independent stochastic decode.

For the current teacher-CE-only run, set:

```bash
ENC_DEC_BEAM_WIDTH=5
ENC_DEC_LENGTH_PENALTY=0.5
```

No adaptive CE/CTC routing is used when launching
`tune_enc_dec_dynamic_eval_teacher_ce.sh`; it defaults to
`--training_mode teacher_ce`. Set `TRAINING_MODE=teacher_kl` to distil the
clean teacher distribution into the augmented student pass with token-level KL.

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

Run the adaptive sweep with the TEDLIUM dev beam-search winner:

```bash
GPU=1 \
ENC_DEC_BEAM_WIDTH=5 \
ENC_DEC_LENGTH_PENALTY=0.5 \
bash launch_scripts/tune_enc_dec_dynamic_eval_adaptive.sh
```

Run the matching no-adapt baseline with the same decode:

```bash
GPU=1 \
ENC_DEC_BEAM_WIDTH=5 \
ENC_DEC_LENGTH_PENALTY=0.5 \
bash launch_scripts/run_enc_dec_dynamic_eval_baseline.sh
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
and hard teacher filters as the adaptive sweep, but defaults to:

```bash
--training_mode teacher_ce
```

The ablation is intended to quantify where plain teacher-forced pseudo-label
adaptation helps, and where it fails under noisier pseudo-labels.

Teacher-KL ablation sweep:

```bash
TRAINING_MODE=teacher_kl \
TEACHER_KL_TEMPERATURE=1.0 \
bash launch_scripts/tune_enc_dec_dynamic_eval_teacher_ce.sh
```

This keeps the same teacher-filtered pseudo-label path but uses the clean
teacher decoder distribution as a soft target for the augmented student pass.

Epoch-relabel teacher ablation:

```bash
TRAINING_MODE=teacher_kl \
TEACHER_EPOCH_RELABEL=1 \
RESULTS_DIR=./results/enc_dec/checkpoint1/enc_dec_teacher_epoch_relabel \
ENC_DEC_BEAM_WIDTH=5 \
ENC_DEC_LENGTH_PENALTY=0.5 \
bash launch_scripts/tune_enc_dec_dynamic_eval_teacher_ce.sh
```

With `TEACHER_EPOCH_RELABEL=1`, each adaptation epoch first labels and filters
all chunks with the epoch-start teacher model. The student then trains for one
epoch on only the retained labels. This keeps the teacher fixed during the
student epoch, then repeats the label/filter/train cycle on the next epoch.
The default chunk-by-chunk teacher/student behavior is unchanged when this
flag is unset.

Current teacher-CE beam sweep:

```bash
GPU=0 \
DATASETS="tedlium earnings22" \
LRS="3e-7 1e-7 1e-6" \
ENC_DEC_BEAM_WIDTH=5 \
ENC_DEC_LENGTH_PENALTY=0.5 \
bash launch_scripts/tune_enc_dec_dynamic_eval_teacher_ce.sh
```

Detached screen form:

```bash
screen -L -Logfile results/enc_dec/checkpoint1/enc_dec_dynamic_eval/teacher_ce_beam5_lp0p5_tedlium_earnings22.log \
  -dmS teacher_ce_beam5_lp0p5_tedlium_earnings22 \
  bash -lc 'GPU=0 DATASETS="tedlium earnings22" LRS="3e-7 1e-7 1e-6" ENC_DEC_BEAM_WIDTH=5 ENC_DEC_LENGTH_PENALTY=0.5 bash launch_scripts/tune_enc_dec_dynamic_eval_teacher_ce.sh'
```

## Files

Pickles are written as:

```text
results/enc_dec/checkpoint1/enc_dec_dynamic_eval/<dataset>-<split>-adaptive_ce_ctc_aux[-beam<beam>_lp<lp>...]-epoch-<E>-lr-<lr_tag>-<aug>-agree<threshold>.pkl
```

Teacher-CE and teacher-KL ablation pickles are written as:

```text
results/enc_dec/checkpoint1/enc_dec_dynamic_eval/<dataset>-<split>-<teacher_ce|teacher_kl>[-beam<beam>_lp<lp>...]-epoch-<E>-lr-<lr_tag>-<aug>.pkl
```

Epoch-relabel variants add `_epoch_relabel` to the mode:

```text
results/enc_dec/checkpoint1/enc_dec_teacher_epoch_relabel/<dataset>-<split>-<teacher_ce_epoch_relabel|teacher_kl_epoch_relabel>[-beam<beam>_lp<lp>...]-epoch-<E>-lr-<lr_tag>-<aug>.pkl
```

For the current beam5/lp0.5 run, expected pickles are:

```text
results/enc_dec/checkpoint1/enc_dec_dynamic_eval/<dataset>-test-teacher_ce-beam5_lp0p5-epoch-1-lr-<lr_tag>-<aug>_1.pkl
```

For example:

```text
results/enc_dec/checkpoint1/enc_dec_dynamic_eval/tedlium-test-teacher_ce-beam5_lp0p5-epoch-1-lr-3em7-freq6_width34_time0_1.pkl
results/enc_dec/checkpoint1/enc_dec_dynamic_eval/earnings22-test-teacher_ce-beam5_lp0p5-epoch-1-lr-1em6-no_aug_1.pkl
```

With repeats, `enc_dec_dynamic_eval_test.py` appends `_<repeat>` before
`.pkl`.

Logs are written under:

```text
results/enc_dec/checkpoint1/enc_dec_dynamic_eval/logs/
```

## Aggregation

Compact table:

```bash
python results/enc_dec/checkpoint1/enc_dec_dynamic_eval/aggregate.py
```

JSON:

```bash
python results/enc_dec/checkpoint1/enc_dec_dynamic_eval/aggregate.py --json
```

CSV:

```bash
python results/enc_dec/checkpoint1/enc_dec_dynamic_eval/aggregate.py --csv results/enc_dec/checkpoint1/enc_dec_dynamic_eval/summary.csv
```
