# enc_dec_v2 checkpoint family

This folder contains results tied to the historical encoder-decoder checkpoint
family:

```text
/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt
```

This is the older checkpoint family that produced the higher Earnings22
unadapted WER in the top-level encoder-decoder outcome history. ROB-63 keeps it
separate from the newer normal seed checkpoint, `old_seed`, and the RL-trained
checkpoint, `rl_step_30000`.

## Contents

| Folder | Meaning |
|---|---|
| `enc_dec_beam_tedlium_dev/` | TED-LIUM dev beam-search decode sweep for this checkpoint family. |
| `enc_dec_dynamic_eval/` | Historical teacher-CE dynamic-eval sweep. |
| `enc_dec_teacher_kl/` | Historical teacher-KL sweep. |
| `enc_dec_teacher_kl_entropy_filter/` | Historical teacher-KL sweep with entropy filtering. |
| `enc_dec_teacher_kl_relaxed_filters/` | Historical teacher-KL sweep with relaxed filters. |
| `enc_dec_teacher_epoch_relabel/` | Historical teacher relabel ablation. |
| `rob63_earnings_unadapted_sanity/` | Direct Earnings22 unadapted sanity check against `old_seed`. |

For ROB-63 seed-vs-RL comparisons, use the generated checkpoint views in
`../old_seed/` and `../rl_step_30000/`.
