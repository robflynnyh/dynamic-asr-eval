# ROB-55 Majority-Vote Stage 1 Snapshot

Date: 2026-05-08

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
