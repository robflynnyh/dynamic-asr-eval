# enc_dec_majority_vote

Encoder-decoder self-training experiments for ROB-55. The core change is the
teacher pseudo-label source: instead of training on a single deterministic
teacher decode, the runner can sample several teacher decodes, select the
largest agreement group, and skip the chunk when that group is too small.

Key CLI controls:

```bash
--teacher_vote_num_samples 8
--teacher_vote_temperature 0.7
--teacher_vote_min_count 3
--teacher_vote_similarity 1.0
```

`teacher_vote_similarity=1.0` requires exact agreement after lowercasing and
whitespace normalization. Lower values use `1 - CER` similarity, which can be
used later if exact voting is too sparse.

## Hypotheses

- Moderate sampling temperature should expose uncertain teacher outputs without
  collapsing to greedy agreement.
- Requiring at least 3 matching predictions out of 8 should remove noisy
  pseudo-labels while retaining enough chunks for one-epoch adaptation.
- `teacher_kl` may be more stable than hard `teacher_ce` when the vote-selected
  transcript is right but still under-specified at token level.
- No-augmentation and weak frequency masking should be tested first because
  teacher forcing may reduce the benefit of aggressive augmentation for
  encoder-decoder adaptation.

## First Bounded Sweep

The initial sweep is intentionally on TEDLIUM dev:

```bash
bash launch_scripts/tune_enc_dec_majority_vote_tedlium_dev.sh
```

Defaults:

| Axis | Values |
|---|---|
| dataset / split | `tedlium / dev` |
| checkpoint | `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt` |
| decode | `beam=5`, `length_penalty=0.5` |
| training modes | `teacher_ce`, `teacher_kl` |
| LRs | `1e-7`, `3e-7` |
| augmentations | `freq3_width24_time0`, `no_aug` |
| vote samples | `8` |
| vote temperatures | `0.7`, `1.0` |
| vote minimum count | `3` |
| vote similarity | `1.0`, `0.9` |
| repeats | `1` |

This is 32 adaptation/eval settings plus one no-adapt baseline. Expand only
after checking retained-label rates and whether the best setting beats the
matching no-adapt baseline on dev.

## Queued Launch

From the repo root, after validating the callback path:

```bash
screen -L -Logfile lcasr/results/enc_dec/enc_dec_majority_vote/logs/screen_rob55_majority_vote_initial.log \
  -dmS rob55_majority_vote_initial \
  bash -lc '/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob55_majority_vote_initial_sweep_queued.sh'
```

The queued wrapper posts a Linear callback on all shell exit paths and moves
ROB-55 back to `Todo` for result inspection.

## Aggregation

```bash
cd lcasr
python results/enc_dec/enc_dec_majority_vote/aggregate.py \
  --csv results/enc_dec/enc_dec_majority_vote/summary.csv
```

## Current Conclusion

The Stage 1b repeat check is recorded in
`../enc_dec_majority_vote_stage1b/OUTCOME.md`. The current majority-vote
teacher formulation should not be expanded as-is: exact voting is too sparse,
and relaxed near-match voting degrades WER.

## Stage 2: Medoid Representative + Confidence Filter

After the 2026-05-09 follow-up request to keep exploring, the next bounded
hypothesis is that relaxed voting may have failed partly because the selected
near-match representative was arbitrary. Stage 2 changes the representative
selection to a medoid candidate: inside the largest support cluster, choose the
sample with the highest mean similarity to the other supported samples.

The first Stage 2 sweep remains on TEDLIUM dev and is intentionally small:

| Axis | Values |
|---|---|
| training modes | `teacher_ce`, `teacher_kl` |
| LRs | `1e-7`, `3e-7` |
| augmentation | `freq3_width24_time0` |
| vote samples / temp / min count | `8` / `0.7` / `3` |
| vote similarities | `0.9`, `0.95` |
| representative | `medoid` |
| confidence filter | mean max prob `>=0.35`, mean entropy `<=2.5` |
| repeats | `2` |

Queued wrapper:

```bash
screen -L -Logfile lcasr/results/enc_dec/enc_dec_majority_vote_stage2_medoid_confidence/logs/screen_rob55_majority_vote_stage2_medoid_confidence.log \
  -dmS rob55_majority_vote_stage2_medoid_confidence \
  bash -lc '/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob55_majority_vote_stage2_medoid_confidence_queued.sh'
```

Expand only if the medoid/confidence run recovers most of the retained-label
rate from relaxed voting without the Stage 1b WER degradation.
