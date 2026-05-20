# RMM Evaluation Results

ROB-68 RMM result artifacts are grouped here to keep the top-level
`lcasr/results/` directory compact.

## Subdirectories

- `ctc_seq2048/`: historical 2048-context RMM run from the initial port.
- `ctc_seq16384/`: corrected normal 16384-context RMM run with fixed
  `time_masks=12`.
- `ctc_seq16384_scaled_time_masks/`: 16384-context follow-up that preserves
  2048-like time-mask widths by scaling the time-mask count.

Each subdirectory owns its raw PKL artifacts, `aggregate.py`, generated summary
tables, and run README.
