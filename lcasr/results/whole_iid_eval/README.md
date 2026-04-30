# whole_iid_eval

Whole-test-set i.i.d. chunk-pool adaptation.

- Launcher: `launch_scripts/tune_whole_iid_eval.sh`
- Runner: `run_whole_iid_eval.py`

## Question

If the model is allowed to adapt on every test recording, does the gain from
`whole_concat_eval` come from seeing all test audio, or from concatenating the
recordings into one long context?

This setup keeps the same adapt/evaluate protocol as `whole_concat_eval`, but
removes cross-recording concatenation. Each recording is chunked independently,
then adaptation samples from the pooled chunks across all recordings. No chunk
crosses a recording boundary.

Like `whole_concat_eval`, this is not an honest generalisation measurement:
the held-out set is still the same recordings used for adaptation. It is a
control for the effect of concatenation.

## Setup

- Load all records for the selected dataset split.
- Build adaptation chunks independently per recording using `SEQ` and
  `ADAPT_OVERLAP`.
- Pool all chunks and shuffle/sample i.i.d. across the pool for each
  adaptation epoch.
- Restore-and-evaluate each original recording with greedy CTC, unless
  `-bs` is set.

## Sweep

The launcher mirrors `whole_concat_eval`:

- Epochs: `{1, 3, 5, 10}`
- LR: `{9e-6, 9e-5}` (`9em6`, `9em5` in filenames)
- Repeats: `3`
- `seq=16384`, `overlap=adapt_overlap=14336`
- Augmentation: `spec_augment_n_freq_masks=6`,
  `spec_augment_freq_mask_param=34`, `spec_augment_n_time_masks=0`

## Running

Default Earnings22 sweep:

```bash
bash launch_scripts/tune_whole_iid_eval.sh
```

TEDLIUM sweep:

```bash
DATASET=tedlium GPU=1 bash launch_scripts/tune_whole_iid_eval.sh
```

Single setting:

```bash
GPU=1 DATASET=earnings22 EPOCHS="10" LRS="9e-6" REPEATS=3 bash launch_scripts/tune_whole_iid_eval.sh
```

## Files

```text
<dataset>-test-whole-iid-epoch-<E>-lr-<lr_tag>_<repeat>.pkl
```

Each pickle stores `baseline`, `adapted`, `delta_wer`,
`baseline_per_record`, `adapted_per_record`, `adapt_num_chunks`,
`chunks_per_record`, `adapt_ids`, and `args_dict`.

`logs/` contains stdout per run.

## Aggregation

```bash
python results/whole_iid_eval/aggregate.py
```

## Plotting

After results exist:

```bash
python results/whole_iid_eval/plot_whole_iid_bars.py --dataset earnings22
python results/whole_iid_eval/plot_whole_iid_bars.py --dataset tedlium --out whole_iid_tedlium_bars.pdf
```
