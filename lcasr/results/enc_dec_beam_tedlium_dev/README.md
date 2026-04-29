# enc_dec_beam_tedlium_dev

Encoder-decoder beam-search decoding sweep on the TEDLIUM dev split.

- Launcher: `launch_scripts/sweep_enc_dec_beam_tedlium_dev.sh`
- Runner: `enc_dec_inference_test.py`
- Checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt`
- Analysis: `OUTCOME.md`

## Question

Can autoregressive encoder-decoder beam search improve TEDLIUM dev WER and
reduce repetition compared with the default greedy encoder-decoder decode?

This sweep keeps beam width fixed at 5 and varies the decoder scoring controls
that seemed most relevant after the first manual tests: length penalty, EOS
bias, repetition penalty, no-repeat n-gram blocking, and a short max-generation
cap.

## Setup

- Dataset: `tedlium`
- Split: `dev`
- Sequence length / overlap: `2048 / 0`
- Flash attention disabled for compatibility: `-dfa`
- Python: `python3.10`
- Beam width: `5` for all beam-search configs

## Sweep

| Run | Extra decoder args |
|---|---|
| `default` | default greedy decoding |
| `beam5_lp0` | `--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0` |
| `beam5_lp0p2` | `--enc_dec_beam_width 5 --enc_dec_length_penalty 0.2` |
| `beam5_lp0p5` | `--enc_dec_beam_width 5 --enc_dec_length_penalty 0.5` |
| `beam5_lp1p0` | `--enc_dec_beam_width 5 --enc_dec_length_penalty 1.0` |
| `beam5_eos0p5` | `--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0 --enc_dec_eos_bias 0.5` |
| `beam5_eos1p0` | `--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0 --enc_dec_eos_bias 1.0` |
| `beam5_rep0p5_ng3` | `--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0 --enc_dec_repetition_penalty 0.5 --enc_dec_no_repeat_ngram_size 3` |
| `beam5_rep1p0_ng3` | `--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0 --enc_dec_repetition_penalty 1.0 --enc_dec_no_repeat_ngram_size 3` |
| `beam5_ng3` | `--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0 --enc_dec_no_repeat_ngram_size 3` |
| `beam5_eos0p5_rep0p5_ng3` | `--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0 --enc_dec_eos_bias 0.5 --enc_dec_repetition_penalty 0.5 --enc_dec_no_repeat_ngram_size 3` |
| `beam5_max80` | `--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0 --enc_dec_max_generate 80` |

## Result

Best WER in this sweep:

```text
beam5_lp0p5: 11.20% WER
```

Near-best alternatives:

```text
beam5_lp0:   11.34% WER
beam5_lp0p2: 11.34% WER
```

Default greedy reference:

```text
default: 12.67% WER
```

See `OUTCOME.md` for the qualitative comparison. The short summary is that
`beam5_lp0p5` wins by WER, while `beam5_lp0` and `beam5_lp0p2` are close and
slightly more conservative. `eos_bias=1.0`, no-repeat n-gram blocking, and
`max_generate=80` were worse in this sweep.

## Running

Default sweep:

```bash
bash launch_scripts/sweep_enc_dec_beam_tedlium_dev.sh
```

Run on a specific GPU:

```bash
GPU=1 bash launch_scripts/sweep_enc_dec_beam_tedlium_dev.sh
```

Print commands without running:

```bash
DRY_RUN=1 bash launch_scripts/sweep_enc_dec_beam_tedlium_dev.sh
```

Run optional Codex post-analysis after the evaluations:

```bash
RUN_CODEX_ANALYSIS=1 bash launch_scripts/sweep_enc_dec_beam_tedlium_dev.sh
```

## Files

Pickles are written as:

```text
results/enc_dec_beam_tedlium_dev/tedlium-dev-<run>-seq2048-overlap0_<repeat>.pkl
```

Logs are written under:

```text
results/enc_dec_beam_tedlium_dev/logs/
```

The detached screen launch log from the first sweep is:

```text
results/enc_dec_beam_tedlium_dev/sweep_screen.log
```

## Aggregation

Compact table:

```bash
python3.10 results/enc_dec_beam_tedlium_dev/aggregate.py
```

JSON:

```bash
python3.10 results/enc_dec_beam_tedlium_dev/aggregate.py --json
```

CSV:

```bash
python3.10 results/enc_dec_beam_tedlium_dev/aggregate.py --csv results/enc_dec_beam_tedlium_dev/summary.csv
```
