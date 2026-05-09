# ROB-57 Entropy Ablation Outcome

Status: queued-run support prepared; full results should be filled in after the detached run completes and the callback returns ROB-57 to `Todo`.

Planned full command:

```bash
/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob57_entropy_ablation_queued.sh
```

The full run uses 5 adaptation epochs on TED-LIUM test and Earnings22 test for `freq_mask` and `no_aug`. The wrapper writes logs under `lcasr/results/entropy_ablation/logs/`, raw entropy traces under `lcasr/results/entropy_ablation/raw/`, normal pickle outputs under `lcasr/results/entropy_ablation/pkl/`, and regenerated aggregate/plot outputs in this directory.
