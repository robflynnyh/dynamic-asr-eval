# gender_eval_tedlium

Cross-speaker-gender dynamic evaluation on TEDLIUM.

- Launcher: `launch_scripts/eval_genders.sh`
- Runner: `run_cross_speaker_gender_tedlium.py`
- Speaker manifest: `speaker_manifest_15x15.json`

## Question
Does test-time adaptation (dynamic eval) on a speaker generalise to other
speakers of the same gender, and how does it transfer across gender?

## Setup
Expanded TEDLIUM speaker pool selected from the train set by:
1. keeping the existing eval women,
2. scanning train talks with transcript span >= 8 minutes,
3. sorting by smallest max transcript gap,
4. taking the first verified female talks until the pool is full.

The current manifest contains:
- 15 female talks
- 15 male talks

For each speaker we adapt the model on that recording (`-seq 16384 -o 14336`),
then evaluate the updated model on:
- the other speakers of the same gender (LOO within gender), and
- every speaker of the opposite gender (cross-gender transfer).

Model weights are restored to the checkpoint between speakers.

## Sweep
- Epochs: 1..5
- LR: 9e-5 (default in the launcher) and 9e-6 (`-lr9e6` filenames)
- Repeats: 3

## Files
`tedlium-epoch-<E>[-lr9e6]-test_<repeat>.pkl`

Each pickle stores per-repeat dicts with:
- `male_baseline`, `female_baseline` — pre-adaptation WER per gender
- `male_to_male`, `female_to_female` — LOO transfer within gender, list per adapt-speaker
- `male_to_female`, `female_to_male` — cross-gender transfer, list per adapt-speaker
- `args_dict` — full launch arguments

`logs/` holds the corresponding stdout from each run.

## Notes
- The manifest is the source of truth for speaker membership and gender.
- Existing legacy 6-speaker runs are still in the repo, but they are not the current selection.

## Headline numbers (from `experiment_outcomes.txt`, lr=9e-6)
- Baselines: male 0.0701, female 0.0409
- Best male→female: epoch 3 (0.04066)
- Best female→male: epoch 1 (0.07027)
