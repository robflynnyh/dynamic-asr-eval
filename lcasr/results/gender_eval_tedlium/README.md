# gender_eval_tedlium

Cross-speaker-gender dynamic evaluation on TEDLIUM.

- Launcher: `launch_scripts/eval_genders.sh`
- Runner: `run_cross_speaker_gender_tedlium.py`

## Question
Does test-time adaptation (dynamic eval) on a speaker generalise to other
speakers of the same gender, and how does it transfer across gender?

## Setup
6 TEDLIUM recordings (test+dev) split by speaker gender:
- Male: `BillGates_2010`, `DanielKahneman_2010`, `TomWujec_2010U`
- Female: `AimeeMullins_2009P`, `JaneMcGonigal_2010`, `ElizabethGilbert_2009`

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

## Headline numbers (from `experiment_outcomes.txt`, lr=9e-6)
- Baselines: male 0.0701, female 0.0409
- Best male→female: epoch 3 (0.04066)
- Best female→male: epoch 1 (0.07027)
