# ROB-55 Stage 6 Deterministic-Anchored Vote Outcome

Date: 2026-05-10

Stage 6 tested whether sampled teacher decodes are more useful as a confidence
gate for the deterministic beam teacher label than as the training label itself.
The run used TEDLIUM dev full recordings, `freq3_width24_time0`, one epoch, one
repeat, `teacher_vote_representative_strategy=deterministic`, and
`teacher_vote_include_deterministic=1`.

Baseline:

- `tedlium-dev-no_adapt-beam5_lp0p5-epoch-0-lr-none-no_aug`
- WER: `0.112026`

Best setting:

- `teacher_ce`, LR `3e-7`, vote `N=16`, temp `0.7`, min count `2`, exact
  similarity `1.0`
- WER: `0.110921`
- Delta vs baseline: `-0.001105` absolute, `-0.99%` relative

Best setting by method family:

| Method | Best WER | Relative delta | Setting |
|---|---:|---:|---|
| `teacher_ce` | `0.110921` | `-0.99%` | LR `3e-7`, `N=16`, temp `0.7`, sim `1.0` |
| `grpo` | `0.111750` | `-0.25%` | LR `1e-7`, `N=16`, temp `1.0`, sim `0.95` |
| `maxrl` | `0.111916` | `-0.10%` | LR `1e-7`, several tied settings |

The relaxed `0.95` deterministic gate was still fragile. Several `0.95`,
temp `0.7` CE settings worsened WER by `+4.29%` to `+6.51%` relative, and the
worst GRPO setting worsened by `+2.76%`. Exact `1.0` support was much safer,
but the one-repeat gain is still far below the issue target of a consistent
`5%` relative improvement.

Interpretation:

- Deterministic anchoring is the first majority-vote variant in this ROB-55
  sequence to show a near-`1%` full-recording TEDLIUM-dev gain.
- The improvement is concentrated in CE; GRPO and MAXRL remained nearly neutral
  under this recipe.
- Relaxed matching should not be expanded further in this formulation. It
  admits too many harmful labels even when the deterministic beam output is the
  final target.

The planned Stage 7 exact-gate CE confirmation sweep was cancelled after
Robert's 2026-05-10 request to end the investigation. The queued screen was
still waiting in `with-gpu`; it had not acquired a GPU, started the wrapper, or
written result logs.

Final interpretation: this is the safest majority-vote variant tested, but it
is still only a one-repeat TEDLIUM-dev gain of about `1%` relative. It should
not be expanded to test sets as-is. Future work should change the
teacher-quality signal or objective rather than keep tuning transcript-string
agreement thresholds.
