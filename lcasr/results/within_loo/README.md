# within_loo

Within-recording leave-one-out (chunk-wise) dynamic eval on earnings22.

- Launcher: `launch_scripts/tune_within_loo.sh`
- Runner: `run_within_recording_loo_eval.py`

## Question
Inside a single long recording, can we get an unbiased estimate of how much
dynamic-eval adaptation helps by adapting on chunk *i* and decoding the
*other* chunks of the same recording?

## Setup
Two-level chunking inside each recording:
- **Outer LOO chunks** (`-loo_s 65536 -loo_o 57344`, stride 8192): the units
  for the leave-one-out loop.
- **Inner windows** (`-seq 16384 -o 14336`): the windows used by
  `dynamic_eval` during adaptation, and by the windowed inference pass.

For each outer chunk *i* with at least one audio-disjoint partner:
1. restore checkpoint weights,
2. run `dynamic_eval` on chunk *i* (returns adapted parameters),
3. with those parameters, run windowed inference on every outer chunk *j*
   whose spec-frame range `[j, j + len_j)` does not intersect chunk *i*'s
   range `[i, i + len_i)` (so the trailer chunk, which may be shorter than
   `loo_seq_len`, is handled with its true length), and accumulate softmax
   probabilities into stitched logits at the correct downsampled positions.

The audio-disjoint requirement is stricter than `j ≠ i` because outer
chunks overlap heavily (default stride = `loo_seq_len − loo_overlap` = 8192,
chunk length = 65536, so adjacent chunks share 87.5% of their audio).
Without the disjoint constraint, decoding chunk *j* with weights adapted on
neighbouring chunk *i* would mostly be decoding audio that the model
already saw during adaptation, defeating the LOO interpretation.

Adapt chunks with no disjoint partner are skipped entirely. Recordings
without any disjoint (*i*, *j*) pair fall back to a no-adapt windowed pass
(`mode = 'fallback_no_disjoint_pairs'`). With the audio-disjoint rule,
positions near the start/end of a recording may end up uncovered; those
positions are dropped before decoding (a warning is printed).

The stitched probabilities are averaged across all *(i, j)* contributions,
log-transformed, and decoded. Recordings with ≤ 1 outer chunk fall back to
baseline windowed inference. A baseline (no-adapt) windowed pass on the full
recording is also computed for direct comparison.

## Sweep
- epoch ∈ {1, 5}
- lr = 9e-5 (default), repeats = 1
- spec_augment_n_freq_masks=6, freq_mask_param=34, n_time_masks=0

## Files
`earnings22-loo65536_57344-inner16384_14336-epoch-<E>-test_<repeat>.pkl`

Each pickle stores:
- `baseline` / `loo` — overall WER dicts (wer, words, ins/del/sub rate)
- `baseline_model_output`, `model_output`, `gold` — per-recording strings
- `per_recording_meta` — list of `{id, n_chunks, mode}` (`mode` is `loo` or
  `fallback_windowed_eval`)
- `args_dict`, `repeat`
