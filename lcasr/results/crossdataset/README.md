# crossdataset

Cross-dataset dynamic-eval transfer. For each ordered pair (A, B) with
A, B in {earnings22, TEDLIUM}, adapt on each recording of A and evaluate on
B (cross-dataset) and on A \ {i} (within-dataset, leave-one-out).
The earnings22 -> TEDLIUM direction is complete; TEDLIUM -> earnings22
is in progress.

- Launchers: `launch_scripts/tune_cross_dataset.sh`,
  `launch_scripts/tune_cross_dataset_epoch1_ao0.sh`
- Runner: `bin/run_cross_dataset_eval.py`

## Question
When dynamic eval is run on a recording from dataset A, do the adapted
weights transfer to a different test set B, or are the gains specific to
the source dataset? How does this compare to leaving one recording out
within A? Run in both directions (A=earnings22, B=TEDLIUM and vice versa)
to check the effect is not asymmetric.

## Setup
For each recording *i* in earnings22:
1. compute baseline (no-adapt) WER on all of earnings22 and on all of TEDLIUM,
2. adapt on recording *i* via `dynamic_eval` (`-seq 16384 -o 14336`,
   `-ao` adapt_overlap),
3. with those adapted weights, decode all of TEDLIUM (`a_to_b`) and the
   remaining earnings22 recordings (`a_to_a_loo`),
4. restore checkpoint weights before the next *i*.

Results are stored as one entry per adapt-recording so per-speaker variance
in `a_to_b` / `a_to_a_loo` is preserved.

## Sweep (repeats = 3)
| Tag | epoch | lr | adapt_overlap |
|---|---|---|---|
| `epoch-1`            | 1 | 9e-5 | 14336 (= overlap) |
| `epoch-1-ao0`        | 1 | 9e-5 | 0 |
| `epoch-1-lr9e6`      | 1 | 9e-6 | 14336 |
| `epoch-5`            | 5 | 9e-5 | 14336 |
| `epoch-5-lr9e6`      | 5 | 9e-6 | 14336 |

## Headline numbers (lr=9e-6, from `experiment_outcomes.txt`)
- earnings22 baseline = 0.18289, tedlium baseline = 0.06227
- Epoch 1: a_to_b = 0.06257, a_to_a_loo = 0.18119
- Epoch 5: a_to_b = 0.06582, a_to_a_loo = 0.18631

Epoch 1 was better than epoch 5 for both cross-dataset transfer and in-dataset
LOO at lr=9e-6.

## Files
`earnings22_tedlium-<tag>_<repeat>.pkl` — each pickle stores:
- `a_baseline`, `b_baseline` — pre-adaptation WER on A and B
- `a_to_b` — list (length |A|) of WER dicts on B after adapting on A[i]
- `a_to_a_loo` — list (length |A|) of WER dicts on A\\{i}
- `dataset_a`, `dataset_b`, `args_dict`, `repeat`

`logs/` — stdout for the runs that wrote one (post-refactor launchers tee
to log files).
`summarize_epoch1_to_latex.py` — helper used to format the epoch-1 numbers.
