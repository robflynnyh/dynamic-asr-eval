# Plan: simplify within-recording LOO

## Resume instructions (read me first next session)
- Working dir: `/exp/exp4/acp21rjf/dynamic-asr-eval/lcasr`.
- Branch: `main`. Current state has the **audio-disjoint** version of the
  LOO runner committed-in-tree but not yet replaced by the simpler scheme
  described below — that rewrite is the open task.
- Key files already touched in the prior session (do not re-derive):
  - `run_within_recording_loo_eval.py` (top-level) — currently implements
    the overlapping-chunks + audio-disjoint variant. To be replaced per
    "New scheme" below.
  - `results/within_loo/README.md` — currently documents the
    audio-disjoint variant. Needs replacement.
  - `results/within_loo/plot_within_loo_scheme.py` — already produces
    a merged-infer diagram (one `infer` block per side of `adapt`), which
    matches the target scheme; minor cleanup only.
- Thesis text in flight (lives outside this repo, in the user's thesis
  source): paragraphs `\paragraph{Cross-Dataset and Within-Dataset}` and
  `\paragraph{Within Recording:}` were drafted this session. NSTI is the
  paper-side name for what the code calls `dynamic_eval`.
- To resume, open this file, then implement the "Files to change" list in
  order. Land it as one commit. Re-running the sweep (`tune_within_loo.sh`)
  is out of scope for the rewrite itself.

## Goal
Replace the current overlapping-outer-chunks + audio-disjoint exclusion
scheme with a much simpler split-then-concat scheme, and update the
runner, README, and diagram to match.

## New scheme
For each recording:
1. Split the spectrogram into `N` **non-overlapping** outer chunks of length
   `loo_seq_len` (last chunk may be shorter). No `loo_overlap` — drop the
   argument or force it to `0`.
2. For each `i` in `0..N-1`:
   a. Restore checkpoint weights.
   b. Adapt on chunk `i` via NSTI (`dynamic_eval`).
   c. Form `infer_before = concat(chunks[0:i])` and
      `infer_after = concat(chunks[i+1:N])`. These are contiguous slices of
      the original recording (no seam, since the chunks are adjacent in the
      source audio). One or both may be empty for `i = 0` / `i = N-1`.
   d. Run `windowed_inference` on `infer_before` and on `infer_after`
      separately. Place each output's softmax probabilities at the
      original spec-frame positions: `infer_before` at position `0`,
      `infer_after` at position `(i + 1) * loo_seq_len` (or the start of
      chunk `i+1`).
   e. Accumulate into stitched logit / count tensors.
3. After all passes, average per position, log, decode.

Each output position is decoded `N - 1` times (once per pass with
`i ≠ its_chunk`), so cross-pass averaging is preserved.

## Files to change
- `run_within_recording_loo_eval.py`
  - Drop `--loo_overlap` (or force `0`).
  - Replace `prepare_chunks` call with a non-overlapping split.
  - Rewrite `loo_eval` per the new scheme above. Delete the
    `disjoint`/`valid_evals`/`fallback_no_disjoint_pairs` machinery and the
    contiguity warning — coverage is by construction now.
  - Update the example invocation in the trailing comment.
- `launch_scripts/tune_within_loo.sh`
  - Drop `LOO_OVERLAP`. Filename tag becomes `loo${LOO_SEQ}` only (or use a
    new tag, e.g. `noverlap-loo${LOO_SEQ}`, so old pickles are
    distinguishable from new ones).
- `results/within_loo/README.md`
  - Replace the audio-disjoint section with a description of the
    non-overlapping split + before/after concat scheme.
  - Note that previously-stored pickles with the old tag are not
    comparable to new ones.
- `results/within_loo/aggregate.py` and
  `results/within_loo/plot_within_loo_bars.py`
  - Update the filename pattern they parse (mainly the dropped
    `loo_overlap` field).
- `results/within_loo/plot_within_loo_scheme.py`
  - Per-pass row becomes exactly: `infer_before` (single box) | `adapt`
    | `infer_after` (single box). Edge passes lose one of the two infer
    boxes. The merged-infer diagram already produced is essentially
    correct; just remove any small gap/asymmetry and update the legend
    text.

## Things to double-check during the rewrite
- `windowed_inference` is happy with arbitrarily long inputs (it should be
  — it already does internal windowing with `-seq` / `-o`). Empty inputs
  for `i = 0` / `i = N-1` need a guard so we just skip that side.
- The downsampled offset placement uses
  `infer_after_start_frame // downsampling_factor`, where
  `infer_after_start_frame` is the spec-frame position of `chunk_{i+1}` in
  the original recording (not within the concatenated tensor).
- Existing pickles in `results/within_loo/*.pkl` were generated under
  earlier schemes; they should be left in place but treated as legacy.

## Out of scope
- Re-running the experiments. Just land the code change and update docs;
  results sweep can follow.
