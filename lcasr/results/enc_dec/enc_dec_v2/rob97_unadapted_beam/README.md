# ROB-97 enc_dec_v2 unadapted beam-search evals

This directory holds the missing before-adaptation rows requested by ROB-97 for
the ROB-96 thesis table.

Fixed decode setting:

```text
beam_width=5
length_penalty=0.5
seq_len=2048
overlap=0
epochs=0
```

Checkpoint:

```text
/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt
```

Target rows:

| Dataset | Split | Purpose |
|---|---|---|
| `chime6` | `test` | Missing enc_dec_v2 before-adaptation CHiME-6 test-set WER for ROB-96. |
| `rev16` | `test` | Missing enc_dec_v2 before-adaptation Rev16 test-set WER for ROB-96. |

Queue from the repository root:

```bash
screen -L -Logfile lcasr/results/enc_dec/enc_dec_v2/rob97_unadapted_beam/screen.log \
  -dmS rob97_encdec_v2_unadapted_beam \
  bash -lc '/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob97_enc_dec_v2_unadapted_beam_queued.sh'
```

The queued wrapper posts a Linear callback to ROB-97 on any exit path. On
success it also notifies ROB-96 and moves ROB-96 to Todo.

Regenerate after PKLs are present from `lcasr/`:

```bash
python results/enc_dec/enc_dec_v2/rob97_unadapted_beam/aggregate.py \
  --csv results/enc_dec/enc_dec_v2/rob97_unadapted_beam/summary.csv \
  --outcome results/enc_dec/enc_dec_v2/rob97_unadapted_beam/OUTCOME.md
```
