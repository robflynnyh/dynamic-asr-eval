# ROB-55 Majority-Vote Outcome

Date: 2026-05-08

## Stage 1 Snapshot

The first TEDLIUM-dev calibration sweep completed all 33 result pickles. The
detached wrapper reported exit status 127 after the final pickle because the
wrapper was patched while the shell process was still live; the model runs
themselves completed and were aggregated afterward.

Baseline:

- `tedlium-dev-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug_1.pkl`
- WER: `0.11203`

Best setting:

- `teacher_ce`, LR `3e-7`, `freq3_width24_time0`, vote `N=8`, temp `0.7`,
  min count `3`, similarity `0.9`
- WER: `0.10970`
- Delta vs baseline: `-0.00232` absolute, `-2.07%` relative

Interpretation:

- The best result is a small positive signal, not the requested 5% relative
  improvement.
- Useful settings were concentrated in CE with moderate frequency masking and
  lower vote temperature.
- Near-match voting with no augmentation was unstable and produced the worst
  WERs.
- KL did not beat CE in this sweep; KL with `similarity=0.9` and augmentation
  was actively worse.

Next step:

- Run a bounded Stage 1b repeat check on TEDLIUM dev before expanding to test
  sets: `teacher_ce`/`teacher_kl`, LR `1e-7`/`3e-7`, `freq3_width24_time0`,
  vote temp `0.7`, min count `3`, similarity `1.0`/`0.9`, `REPEATS=3`.

## Stage 1b Repeat Check

The Stage 1b queued repeat check completed successfully with callback exit
status `0` and wrote
`../enc_dec_majority_vote_stage1b/summary.csv`.

Baseline:

- WER: `0.112026`

Best repeat-mean setting:

- `teacher_kl`, LR `3e-7`, `freq3_width24_time0`, vote `N=8`, temp `0.7`,
  min count `3`, exact similarity `1.0`
- WER: `0.111566`
- Delta vs baseline: `-0.000461` absolute, `-0.41%` relative

Relaxed `0.9` similarity was consistently worse than baseline, with mean WERs
from `0.118584` to `0.120980` (`+5.85%` to `+7.99%` relative degradation).

Vote diagnostics from the Stage 1b logs:

- Exact similarity `1.0` accepted only `22-23` vote updates over three repeats
  and skipped `892-893` updates as below threshold.
- Relaxed similarity `0.9` accepted `558-582` vote updates but degraded WER,
  indicating that near-match pseudo-label quality is too low.

## Conclusion

This majority-vote teacher formulation is not a good expansion target. Exact
voting is safe but too sparse to produce meaningful adaptation, while relaxed
near-match voting supplies enough labels but harms WER. The observed best
repeat-mean gain is `0.41%` relative on TEDLIUM dev, far below the target
`5%` relative gain. Further work should change the agreement primitive or add a
stronger confidence filter before spending GPU time on test-set or GRPO/MAXRL
expansion.
