# TEDLIUM dev encoder-decoder beam refinement outcome

Artifacts inspected:

- Pickles: `./results/enc_dec_beam_tedlium_dev_refine/*_1.pkl`
- Logs: `./results/enc_dec_beam_tedlium_dev_refine/logs/*.log`
- Baseline summary: `./results/enc_dec_beam_tedlium_dev/summary.csv`

No evaluations were rerun. Metrics below come from saved pickle fields
`wer`, `ins_rate`, `del_rate`, and `sub_rate`; saved `args_dict` values and
the sweep log were used to recover decoder arguments. Each refinement run
covers 8 TEDLIUM dev talks and 18,094 reference words.

## Refinement Results

| Run | Arguments | WER | Ins | Del | Sub | Notes |
|---|---|---:|---:|---:|---:|---|
| `tedlium-dev-beam10_lp0p5-seq2048-overlap0` | `beam=10, lp=0.5, no_repeat_ngram=0` | **11.15%** | 1.48% | 3.48% | 6.19% | Best WER in this refinement. Output/reference ratio 0.980. Similar repetition profile to prior beam-5 baseline, with max repeated 8-gram count 2. |
| `tedlium-dev-beam5_lp0p5_ng8-seq2048-overlap0` | `beam=5, lp=0.5, no_repeat_ngram=8` | 11.25% | 1.49% | 3.51% | 6.25% | Close to prior beam-5 baseline, but +0.04 absolute WER. Avoids the beam-3 loop, but final concatenated output still has a repeated 8-gram count of 2. |
| `tedlium-dev-beam3_lp0p5-seq2048-overlap0` | `beam=3, lp=0.5, no_repeat_ngram=0` | 11.87% | 2.16% | 3.42% | 6.28% | Worse than beam 5 and beam 10. Higher insertion rate and an obvious repeated phrase loop: max repeated 8-gram count 21. |

## Baseline Comparison

From `results/enc_dec_beam_tedlium_dev/summary.csv`, the strongest previous
setting was `tedlium-dev-beam5_lp0p5-seq2048-overlap0`:

| Run | Arguments | WER | Ins | Del | Sub | Notes |
|---|---|---:|---:|---:|---:|---|
| `tedlium-dev-beam5_lp0p5-seq2048-overlap0` | `beam=5, lp=0.5, no_repeat_ngram=0` | 11.20% | 1.53% | 3.44% | 6.23% | Previous best. Output/reference ratio 0.981; max repeated 8-gram count 2. |
| `tedlium-dev-default-seq2048-overlap0` | default/greedy | 12.67% | 2.42% | 3.63% | 6.63% | Much worse than beam search and has the known long repetition loop. |
| `tedlium-dev-beam5_ng3-seq2048-overlap0` | `beam=5, lp=0.0, no_repeat_ngram=3` | 13.00% | 1.59% | 4.01% | 7.40% | Stronger local n-gram blocking, but too much WER degradation. |

Relative to the prior `beam=5, lp=0.5` baseline, `beam=10, lp=0.5` improves
WER by 0.050 absolute percentage points, mostly through slightly lower
insertion and substitution rates. This is a real win in the saved metrics, but
small enough that it should be treated as a tie unless runtime cost is
acceptable.

`beam=3, lp=0.5` is not competitive. Its WER is 0.66 absolute percentage
points worse than the prior beam-5 baseline and 0.71 points worse than beam 10.
The regression is concentrated in insertions, and the output repeats
`and i also think it is dangerous and` 21 times in one talk, matching the class
of failure seen in greedy decoding.

## No-repeat Ngram 8

`no_repeat_ngram_size=8` looks safe in the narrow sense that it does not cause
the large WER penalty seen with the earlier `no_repeat_ngram_size=3` sweep.
The WER moves from 11.20% to 11.25% against the matching `beam=5, lp=0.5`
baseline, with very similar insertion, deletion, substitution, and length-ratio
behavior.

It is not a complete global repetition guard in the saved final text. The final
concatenated model output still contains a repeated 8-gram count of 2. That may
come from independent chunk decoding or natural repeated phrasing, but the
important practical result is that it prevents the beam-3/default style
long-loop failure without materially damaging WER.

## Recommendation

Use `beam=5, lp=0.5` as the default quality/runtime point. Add
`no_repeat_ngram_size=8` when a cheap repetition guard is worth a negligible WER
cost. Use `beam=10, lp=0.5` only if the extra decoding cost is acceptable for a
small 0.05-point WER gain. Do not use `beam=3` for this setting.
