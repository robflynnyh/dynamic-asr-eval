# CTC 8192-Context Self-Training Eval

ROB-115 adds the missing adapted 8192-context CTC evaluation requested after
ROB-67. This run intentionally covers the adapted/frequency-masking row only;
the 8192 no-adapt baseline is not duplicated here.

## Setup

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_8192_rp_1/step_105360.pt`
- Sequence length: `8192`
- Overlap: `7168` (`8192 * 0.875`, stride `1024`)
- Datasets: `earnings22`, `tedlium`, `chime6`, `rev16`
- Split: `test`
- Adaptation epochs: `5`
- LRs: `9e-5` and matched follow-up `6e-5`
- Augmentation: `spec_augment_n_freq_masks=6`, `spec_augment_freq_mask_param=34`, `spec_augment_n_time_masks=0`
- Repeats: `1`

## Launch

From the repo root, the callback-backed Mimas launch is:

```bash
screen -L -Logfile lcasr/results/ctc_seq8192_self_training_eval/logs/rob115_ctc_seq8192_self_training.screen.log \
  -dmS rob115_ctc_seq8192_self_training \
  bash -lc '/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob115_ctc_seq8192_self_training_queued.sh'
```

The matched `6e-5` follow-up used the same wrapper with `LR=6e-5`, a distinct
screen name, and the same result directory:

```bash
screen -L -Logfile lcasr/results/ctc_seq8192_self_training_eval/logs/rob115_ctc_seq8192_self_training_lr6em5.screen.log \
  -dmS rob115_ctc_seq8192_self_training_lr6em5 \
  bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-115 && LR=6e-5 SCREEN_NAME=rob115_ctc_seq8192_self_training_lr6em5 LOG_PATH=/exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-115/lcasr/results/ctc_seq8192_self_training_eval/logs/rob115_ctc_seq8192_self_training_lr6em5.log QUEUED_COMMAND="LR=6e-5 /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob115_ctc_seq8192_self_training_queued.sh" /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob115_ctc_seq8192_self_training_queued.sh'
```

The wrapper sets `TMPDIR`, `TEMP`, `TMP`, `MPLCONFIGDIR`, `HF_HOME`, and
`XDG_CACHE_HOME` under `symphony/.scratch/ROB-115/`.

For command-shape validation from `lcasr/`:

```bash
DRY_RUN=1 DATASETS=tedlium EPOCHS=5 MAX_RECORDS=1 \
  bash launch_scripts/run_ctc_seq8192_self_training_eval.sh
```

## Outputs

Each completed cell writes:

```text
<dataset>-test-ctc-seq8192-overlap7168-epoch-5-lr-<lr>_<repeat>.pkl
```

The comparison snapshot reads the 8192 and 16384 no-adapt summaries imported
from the already generated ROB-110 result branch; those baselines were not
rerun for ROB-115. The snapshot includes both completed 8192 adapted learning
rates plus the committed 2048, 16384, and 65536 comparison rows available in
this checkout.

Regenerate summaries from artifacts:

```bash
python lcasr/results/ctc_seq8192_self_training_eval/aggregate.py
python lcasr/results/ctc_seq8192_self_training_eval/compare_snapshot.py
```

`comparison_snapshot.md` remains labeled as a snapshot because it combines
single-repeat rows from multiple committed result families and, for some
long-context adapted comparisons, different adaptation policies.
