# TEDLIUM dev encoder-decoder beam-search sweep outcome

## Current aggregate snapshot

Generated from `aggregate.py` over the current pickles in this directory.
All rows have `n=1`, so the table is a single-run snapshot rather than a
repeat-averaged estimate. This folder intentionally mixes greedy/default and
beam-search decoding settings; WER differences here are decode-setting
differences, not adaptation deltas.

| Rank | Setting | WER | Ins | Del | Sub | Note |
|---:|---|---:|---:|---:|---:|---|
| 1 | `tedlium-dev-beam10_lp0p5-seq2048-overlap0` | 11.15 | 1.48 | 3.48 | 6.19 | Best current WER; small gain over beam 5 at higher decode cost. |
| 2 | `tedlium-dev-beam5_lp0p5-seq2048-overlap0` | 11.20 | 1.53 | 3.44 | 6.23 | Best original sweep setting and preferred runtime-quality point. |
| 3 | `tedlium-dev-beam5_lp0p5_ng8-seq2048-overlap0` | 11.25 | 1.49 | 3.51 | 6.25 | Near tie; adds a light no-repeat guard. |
| 4 | `tedlium-dev-beam5_lp0-seq2048-overlap0` | 11.34 | 1.52 | 3.63 | 6.18 | Conservative near-best. |
| 5 | `tedlium-dev-beam5_lp0p2-seq2048-overlap0` | 11.34 | 1.51 | 3.61 | 6.21 | Conservative near-best. |

Worst settings in the current aggregate are `beam5_max80` at 18.10% WER,
`beam5_rep1p0_ng3` at 13.58%, and `beam5_eos0p5_rep0p5_ng3` at 13.34%.
The `max_generate=80` setting fails by deletion/truncation, while the
no-repeat/repetition-penalty variants are worse mostly through deletion and
substitution increases.

Recommendation: use `beam=5`, `length_penalty=0.5` as the default
adaptation decode setting. `beam=10`, `length_penalty=0.5` is the best saved
WER, but the 0.05 absolute WER gain is small enough to treat as a runtime-cost
tradeoff.

---

Artifacts inspected:

- Pickles: `./results/enc_dec/enc_dec_v2/enc_dec_beam_tedlium_dev/*_1.pkl`
- Logs: `./results/enc_dec/enc_dec_v2/enc_dec_beam_tedlium_dev/logs/*.log`
- No evaluations were rerun. Metrics below come from the saved pickle fields `wer`, `ins_rate`, `del_rate`, and `sub_rate`; logs were used to recover run arguments and completion status.

The completed runs cover 8 TEDLIUM dev talks and 18,094 reference words. All 12 configs from `launch_scripts/sweep_enc_dec_beam_tedlium_dev.sh` completed and produced matching pickle files.

## Summary table

| Run | Arguments | WER | Ins | Del | Sub | Notes |
|---|---:|---:|---:|---:|---:|---|
| `tedlium-dev-beam5_lp0p5-seq2048-overlap0` | `beam=5, lp=0.5` | **11.20%** | 1.53% | 3.44% | 6.23% | Best WER. No obvious truncation; output/reference length ratio 0.981. |
| `tedlium-dev-beam5_lp0-seq2048-overlap0` | `beam=5, lp=0.0` | 11.34% | 1.52% | 3.63% | 6.18% | Near tie with best. Slightly lower insertion/repetition tendency, slightly more deletion. |
| `tedlium-dev-beam5_lp0p2-seq2048-overlap0` | `beam=5, lp=0.2` | 11.34% | 1.51% | 3.61% | 6.21% | Near identical to `lp=0.0`; output/reference length ratio 0.979. |
| `tedlium-dev-beam5_eos0p5-seq2048-overlap0` | `beam=5, lp=0.0, eos_bias=0.5` | 11.39% | 1.51% | 3.70% | 6.18% | Competitive, but slightly shorter and more deletion than the best cluster. |
| `tedlium-dev-beam5_lp1p0-seq2048-overlap0` | `beam=5, lp=1.0` | 11.48% | 1.96% | 3.30% | 6.22% | Longer outputs and more insertions; retains a visible repeated phrase on Elizabeth Gilbert. |
| `tedlium-dev-beam5_eos1p0-seq2048-overlap0` | `beam=5, lp=0.0, eos_bias=1.0` | 12.38% | 1.48% | 4.82% | 6.07% | Lower insertion but clear early-stop risk: output/reference ratio 0.967, min talk ratio 0.925. |
| `tedlium-dev-default-seq2048-overlap0` | default/greedy | 12.67% | 2.42% | 3.63% | 6.63% | Worse than beam search; one obvious repetition loop repeats "and i also think it is dangerous" 21 times. |
| `tedlium-dev-beam5_ng3-seq2048-overlap0` | `beam=5, lp=0.0, no_repeat_ngram=3` | 13.00% | 1.59% | 4.01% | 7.40% | Reduces repeated n-grams, but substitutions/deletions rise; some awkward local word choices. |
| `tedlium-dev-beam5_rep0p5_ng3-seq2048-overlap0` | `beam=5, lp=0.0, rep=0.5, no_repeat_ngram=3` | 13.28% | 1.51% | 4.34% | 7.42% | Repetition safer than default, but noticeably worse WER and shorter outputs. |
| `tedlium-dev-beam5_eos0p5_rep0p5_ng3-seq2048-overlap0` | `beam=5, lp=0.0, eos_bias=0.5, rep=0.5, no_repeat_ngram=3` | 13.34% | 1.49% | 4.43% | 7.42% | Combines no-repeat with EOS bias; highest deletion in this family. |
| `tedlium-dev-beam5_rep1p0_ng3-seq2048-overlap0` | `beam=5, lp=0.0, rep=1.0, no_repeat_ngram=3` | 13.58% | 1.37% | 4.52% | 7.69% | Lowest insertion rate, but too deletion/substitution heavy. |
| `tedlium-dev-beam5_max80-seq2048-overlap0` | `beam=5, lp=0.0, max_generate=80` | 18.10% | 1.33% | 10.88% | 5.89% | Clear truncation/length cap failure; output/reference ratio 0.905, min talk ratio 0.841. |
## Qualitative comparison

The main beam-search cluster, `beam=5` with `lp=0.0`, `0.2`, or `0.5`, is stable. These outputs are very similar to each other, preserve the end of all 8 talks, and avoid the severe default greedy repetition loop. The best run, `lp=0.5`, is slightly longer than `lp=0.0/0.2` and wins by deletion reduction, not by a large qualitative shift.

The default/greedy run has the most obvious qualitative failure. In the Elizabeth Gilbert talk it gets stuck repeating "and i also think it is dangerous" 21 times before recovering. The `beam=5, lp=0.5` output still repeats the phrase twice, but it does not loop and continues with the intended content. This is the clearest qualitative improvement from beam search.

The higher length penalty, `lp=1.0`, is not safer despite a competitive WER. It raises insertions to 1.96% and repeats the same Elizabeth Gilbert phrase 12 times, suggesting the added length reward can reintroduce loop-like behavior.

EOS bias shows the expected tradeoff. `eos_bias=0.5` remains competitive but is slightly shorter and more deletion-prone than `lp=0.0/0.2/0.5`. `eos_bias=1.0` is too aggressive: deletion rises to 4.82%, and several talks become materially short, especially Blaise Aguera y Arcas at 0.925 output/reference length.

The no-repeat/repetition-penalty runs reduce repeated n-gram counts and eliminate the long greedy loop, but they are not better ASR outputs here. WER increases to 13.00-13.58%, mostly through substitutions and deletions. Spot checks show local awkwardness such as `go fire quickly` for `go far quickly` and `known you worlds` for `no new worlds`.

`max_generate=80` is a clear truncation setting. It has the lowest insertion rate but deletion jumps to 10.88%, and every talk is shorter than reference. It should not be used as-is for this setup.

## Best and safest settings

Best by WER: `tedlium-dev-beam5_lp0p5-seq2048-overlap0` (`beam=5`, `enc_dec_length_penalty=0.5`) at 11.20% WER.

Qualitatively safest near-best alternative: `tedlium-dev-beam5_lp0-seq2048-overlap0` or `tedlium-dev-beam5_lp0p2-seq2048-overlap0`. They are only 0.13 absolute WER worse than the best run, have marginally lower insertion/repeated-n-gram tendency, and avoid the `lp=1.0` repetition regression and the EOS early-stop risk.

If repetition avoidance is the overriding goal, `tedlium-dev-beam5_ng3-seq2048-overlap0` is safer than default against loops, but its WER penalty is too large to choose as the primary setting from this sweep.

## Recommendation for next sweep

Center the next sweep on `beam=5` with length penalty in the narrow range around the winner: `lp=0.3, 0.4, 0.5, 0.6, 0.7`. Include `beam=4` and `beam=6` at `lp=0.4-0.6` if runtime allows. Avoid `max_generate=80`, avoid `eos_bias=1.0`, and only retest `no_repeat_ngram=3` with softer settings if loop prevention is more important than the current WER target.
