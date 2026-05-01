# Encoder-Decoder VAD Chunking Plan

## Goal

Add an opt-in encoder-decoder chunking mode that uses voice activity detection
to create speech-centered chunks instead of fixed `seq_len` windows. Fixed
chunking must remain the default so existing experiments and launchers are not
changed.

Primary motivation: test whether encoder-decoder adaptation and decoding improve
when chunk boundaries avoid silence and long arbitrary cuts.

## Current State

Encoder-decoder code currently uses fixed spectrogram chunks:

- `lib.prepare_chunks(spec, seq_len, overlap)` creates windows at
  `range(0, spec_n, seq_len - overlap)`.
- Encoder-decoder paths assert `overlap == 0`.
- Main call sites:
  - `lib.enc_dec_dynamic_eval`
  - `lib.enc_dec_inference`
  - `lib.enc_dec_beamsearch_inference`
  - `lib.enc_dec_ctc_beamsearch_inference`
- `enc_dec_dynamic_eval_test.py` gets each record from the dataset loaders,
  calls `process_fn`, and passes only the processed spectrogram into
  `enc_dec_dynamic_eval`.

VAD libraries such as WebRTC VAD operate on waveform PCM, not on the processed
spectrogram. So the runner needs access to the raw audio path in addition to the
spectrogram.

## Proposed CLI

Add these flags to `enc_dec_dynamic_eval_test.py`:

```bash
--enc_dec_chunking fixed|vad
--vad_aggressiveness 2
--vad_frame_ms 30
--vad_min_speech_ms 250
--vad_merge_gap_ms 300
--vad_pad_ms 200
--vad_max_segment_frames 2048
--vad_min_segment_frames 128
--vad_drop_short_segments
```

Defaults:

- `--enc_dec_chunking fixed`
- VAD flags are ignored unless `--enc_dec_chunking vad`.

Use a separate prefix from decode flags so result metadata can distinguish
chunking from beam-search settings.

## Implementation Steps

1. Add a small VAD utility module.

   Suggested file:

   ```text
   enc_dec_vad_chunks.py
   ```

   Responsibilities:

   - Load raw audio as mono 16 kHz PCM.
   - Run WebRTC VAD on 10/20/30 ms frames.
   - Convert voiced frames into speech spans in seconds.
   - Merge nearby spans using `vad_merge_gap_ms`.
   - Pad spans using `vad_pad_ms`.
   - Convert seconds to spectrogram frame indices.
   - Split segments longer than `vad_max_segment_frames`.
   - Drop or merge segments shorter than `vad_min_segment_frames`.
   - Return:

     ```python
     chunks: dict[int, torch.Tensor]
     keys: list[int]
     metadata: dict
     ```

2. Add a generic chunk-preparation wrapper in `lib.py`.

   Suggested API:

   ```python
   def prepare_enc_dec_chunks(args, spec, seq_len, overlap, record=None):
       if args.enc_dec_chunking == "fixed":
           chunks, keys = prepare_chunks(spec, seq_len, overlap)
           return chunks, keys, {"chunking": "fixed"}
       if args.enc_dec_chunking == "vad":
           return prepare_vad_chunks(...)
   ```

   This keeps `prepare_chunks` unchanged and reduces risk to existing CTC and
   fixed enc-dec experiments.

3. Pass record metadata into encoder-decoder dynamic eval.

   Current call:

   ```python
   enc_dec_dynamic_eval(args, model, spec=audio_spec, ...)
   ```

   Proposed:

   ```python
   enc_dec_dynamic_eval(args, model, spec=audio_spec, ..., record=data[rec])
   ```

   The record contains `audio`, which is the raw audio path in TEDLIUM,
   Earnings22, CHiME-6, and Rev16 loaders.

4. Replace fixed chunk calls in encoder-decoder paths.

   Use the wrapper in:

   - `enc_dec_dynamic_eval`
   - `enc_dec_inference`
   - `enc_dec_beamsearch_inference`
   - `enc_dec_ctc_beamsearch_inference`

   Keep fixed behavior byte-for-byte equivalent when `enc_dec_chunking=fixed`.

5. Store chunking metadata in result pickles.

   Add fields like:

   ```python
   "enc_dec_chunking": "vad",
   "vad_num_chunks": ...,
   "vad_total_speech_seconds": ...,
   "vad_segments": [...],
   "vad_dropped_segments": ...,
   "vad_merged_segments": ...,
   "vad_split_segments": ...,
   ```

   For fixed mode, store at least:

   ```python
   "enc_dec_chunking": "fixed",
   "num_chunks": ...
   ```

6. Add launch/test commands.

   Quick no-adaptation sanity check:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python3.10 enc_dec_dynamic_eval_test.py \
     --dataset tedlium --split dev --breaks \
     --training_mode teacher_kl \
     --enc_dec_chunking vad \
     --vad_aggressiveness 2 \
     --vad_frame_ms 30 \
     --vad_merge_gap_ms 300 \
     --vad_pad_ms 200 \
     --vad_max_segment_frames 2048 \
     -epochs 0 \
     -seq 2048 -o 0 \
     --enc_dec_beam_width 5 \
     --enc_dec_length_penalty 0.5 \
     -c /store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt \
     -dfa \
     -kwargs optim_lr=0.0 spec_augment_n_freq_masks=0 spec_augment_n_time_masks=0
   ```

   Then compare against the same command with:

   ```bash
   --enc_dec_chunking fixed
   ```

## Validation Plan

1. Unit-level checks.

   - VAD span conversion never produces negative frame indices.
   - Segment end is always greater than start.
   - Chunks never exceed `vad_max_segment_frames`.
   - Fixed chunking output is unchanged.

2. Dry-run metadata check.

   Run one TEDLIUM dev recording with `--breaks`.

   Confirm logs show:

   - number of raw VAD spans,
   - number after merge/pad/split,
   - min/mean/max chunk length,
   - total covered speech seconds.

3. No-adaptation WER check.

   Before testing adaptation, compare:

   - fixed chunking, beam5/lp0.5, no adaptation,
   - VAD chunking, beam5/lp0.5, no adaptation.

   If VAD chunking hurts no-adapt WER substantially, tune VAD merge/padding
   before adaptation.

4. Adaptation check.

   Only after no-adapt behavior is acceptable:

   - run `teacher_kl + --teacher_epoch_relabel`,
   - use TEDLIUM dev,
   - keep the current best fixed-chunk decode args,
   - compare VAD vs fixed using the same LR/KL/augmentation/filter setting.

## Risks

- WebRTC VAD false negatives may drop speech and hurt WER.
- Too many short chunks can make encoder-decoder generation unstable.
- Long speech regions still require splitting, so VAD is not a full replacement
  for max-length chunking.
- Chunking changes both adaptation examples and final decoding examples, so
  no-adapt VAD baselines are required.
- Dataset preprocessing may zero out ignored TEDLIUM regions after spectrogram
  creation; VAD over raw audio will not know about those ignored scoring spans
  unless we explicitly account for them.

## Recommended First Pass

Start with TEDLIUM dev only:

1. Implement fixed/vad switch.
2. Run `--breaks` no-adapt debug on one file.
3. Run full TEDLIUM dev no-adapt fixed vs VAD.
4. If no-adapt VAD is close or better, run a small teacher-KL adaptation
   comparison.

Do not launch a broad adaptation sweep until the no-adapt VAD baseline is
understood.
