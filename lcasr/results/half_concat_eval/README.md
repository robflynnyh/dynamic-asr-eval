# half_concat_eval

Within-dataset half-split adaptation on earnings22.

- Launcher: `launch_scripts/tune_half_concat_eval.sh`
- Runner: `run_half_concat_eval.py`

## Question
If we have access to a chunk of in-domain data at test time, can a single
adapt-only pass over its concatenated spectrogram lower WER on the held-out
half of the same dataset?

## Setup
The 6 earnings22 test recordings are split into two halves (deterministic
order, optional `--shuffle_splits`). One half is concatenated along the time
axis into a single spectrogram and used for adaptation only — the stitched
logits are not built (`adapt_on_concat_only`). The held-out half is then
transcribed with the adapted weights. Halves are not swapped within a single
pickle; repeats re-run the same split with fresh randomness in the adapt loop.

Each pickle records `baseline` and `adapted` WER on the held-out half plus
per-record predictions.

## Sweep (epoch = 1, repeats = 3)
| Tag | adapt_overlap | optim_lr |
|---|---|---|
| `epoch-1` (Run A)             | 14336 | 9e-5 |
| `epoch-1-ao0-lr9e6` (Run B)   | 0     | 9e-6 |
| `epoch-1-ao14336-lr9e6` (Run C) | 14336 | 9e-6 |
| `epoch-1-ao14336-lr1e6` (Run D) | 14336 | 1e-6 |

Eval overlap is fixed at 14336.

## Headline numbers (mean over 3 repeats)
Baseline: 0.18347.

| Run | Adapted WER | Δ vs baseline |
|---|---|---|
| C (ao=14336, lr=9e-6) | 0.17877 | −0.00469 |
| D (ao=14336, lr=1e-6) | 0.17937 | −0.00410 |
| B (ao=0,     lr=9e-6) | 0.18129 | −0.00218 |
| A (ao=14336, lr=9e-5) | 0.18902 | +0.00556 |

Lowering LR from 9e-5 → 9e-6 was the dominant effect; matched adapt_overlap
beat ao=0 at the lower LR. See `experiment_outcomes.txt` for the full write-up.

## Files
`earnings22-test-half-concat-<tag>_<repeat>.pkl` — pickled dict per repeat
with `baseline`, `adapted`, `delta_wer`, `baseline_per_record`,
`adapted_per_record`, `concat_spec_shape`, `args_dict`, etc.
`logs/` — stdout per run. `experiment_outcomes.txt` — analysis log.
