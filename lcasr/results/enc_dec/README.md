# Encoder-decoder results

This folder groups the encoder-decoder decoding and test-time adaptation result
sets.

## Layout

| Folder | Purpose |
|---|---|
| `enc_dec_beam_tedlium_dev` | TEDLIUM dev beam-search decode sweep used to choose `beam=5`, `length_penalty=0.5` as the default quality/runtime decode. |
| `enc_dec_dynamic_eval` | Teacher-CE dynamic-eval sweep with beam5/lp0.5 teacher and final decoding, including same-decode no-adapt baselines. |
| `enc_dec_teacher_kl` | Teacher-KL dynamic-eval sweep with the original teacher filters. |
| `enc_dec_teacher_kl_entropy_filter` | Teacher-KL sweep with additional low-confidence entropy filtering. |
| `enc_dec_teacher_kl_relaxed_filters` | Teacher-KL sweep with relaxed repeated-token filtering and no CTC agreement filter. |
| `enc_dec_teacher_epoch_relabel` | Opt-in CE/KL teacher relabel ablation where each epoch labels and filters all chunks before student training. |

Each child folder contains:

```text
README.md
OUTCOME.md
aggregate.py
logs/
```

## Decode Comparability

Some folders evaluate greedy/default decoding, while others evaluate
beam-search decoding. Baselines are only comparable when the decode condition
matches.

The current adaptation folders mostly use:

```text
beam_width=5
length_penalty=0.5
```

The beam sweep folder intentionally compares many decode settings. Its WERs
measure decode-setting effects, not adaptation effects.

## Aggregation

Run any child aggregate from the repo root:

```bash
python results/enc_dec/enc_dec_beam_tedlium_dev/aggregate.py
python results/enc_dec/enc_dec_dynamic_eval/aggregate.py
python results/enc_dec/enc_dec_teacher_kl/aggregate.py
python results/enc_dec/enc_dec_teacher_kl_entropy_filter/aggregate.py
python results/enc_dec/enc_dec_teacher_kl_relaxed_filters/aggregate.py
python results/enc_dec/enc_dec_teacher_epoch_relabel/aggregate.py
```

To refresh CSV summaries:

```bash
python results/enc_dec/enc_dec_dynamic_eval/aggregate.py --csv results/enc_dec/enc_dec_dynamic_eval/summary.csv
python results/enc_dec/enc_dec_teacher_kl/aggregate.py --csv results/enc_dec/enc_dec_teacher_kl/summary.csv
python results/enc_dec/enc_dec_teacher_kl_entropy_filter/aggregate.py --csv results/enc_dec/enc_dec_teacher_kl_entropy_filter/summary.csv
python results/enc_dec/enc_dec_teacher_kl_relaxed_filters/aggregate.py --csv results/enc_dec/enc_dec_teacher_kl_relaxed_filters/summary.csv
python results/enc_dec/enc_dec_teacher_epoch_relabel/aggregate.py --csv results/enc_dec/enc_dec_teacher_epoch_relabel/summary.csv
```

## Launch Notes

Launch commands inside the child READMEs now point at nested
`results/enc_dec/...` output folders. For new runs, keep `RESULTS_DIR` under
this parent folder when the launcher supports it.
