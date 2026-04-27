# whole_concat_eval

Whole-test-set concatenation adaptation on earnings22.

- Launcher: `launch_scripts/tune_whole_concat_eval.sh`
- Runner: `run_whole_concat_eval.py`

## Question
If we concatenate the *entire* test set into one long spectrogram, run an
adapt-only pass for N epochs over it, and then re-transcribe every recording
with the adapted model, does that beat the baseline? How does this scale with
epochs and learning rate?

This is the "use everything you have at test time" upper-bound counterpart to
`half_concat_eval`. The held-out set is the same recordings used for adapt,
so it is *not* an honest generalisation measurement — it bounds how much
adaptation can help if the test recordings themselves were the adaptation
data.

## Setup
- All earnings22 test recordings → concatenate along time axis.
- One adapt-only pass (`adapt_on_concat_only`) for N epochs at LR `lr`.
- Restore-and-evaluate each original recording with greedy CTC (and optional
  beam search if `-bs` is set; it is off in the launcher).

## Sweep (repeats = 3)
- Epochs: {1, 3, 5, 10}
- LR: {9e-6, 9e-5} (LR tag uses `m` in place of `-`, e.g. `9em6`)
- seq=16384, overlap=adapt_overlap=14336

## Files
`earnings22-test-whole-concat-epoch-<E>-lr-<lr_tag>_<repeat>.pkl`

Each pickle stores `baseline`, `adapted`, `delta_wer`,
`baseline_per_record`, `adapted_per_record`, `concat_spec_shape`,
`adapt_ids`, `args_dict`.

`logs/` — stdout per run.
