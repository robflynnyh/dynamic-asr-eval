# ROB-57 Entropy Ablation Outcome

Status: complete. The detached full run exited with status 0 and regenerated the raw traces, aggregate tables, plots, and normal dynamic-eval pickle outputs.

Full queued command:

```bash
screen -L -Logfile lcasr/results/entropy_ablation/logs/rob57_entropy_ablation.screen.log -dmS rob57_entropy_ablation bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-57 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob57_entropy_ablation_queued.sh'
```

The full run used 5 adaptation epochs on TED-LIUM test and Earnings22 test for `freq_mask` and `no_aug`. Each raw trace records `pre_update` entropy before an optimizer step and `post_update` entropy after the step; the plotted x-axis is completed adaptation updates.

## Outputs

- Raw traces: `raw/{tedlium,earnings22}-test-epoch-5-{freq_mask,no_aug}.jsonl`
- Aggregates: `trace_rows.csv`, `entropy_by_update.csv`
- Plots: `entropy_by_update.pdf`, `entropy_by_update.png`
- Evaluation artifacts: `pkl/{tedlium,earnings22}-test-epoch-5-{freq_mask,no_aug}_1.pkl`
- Logs are retained locally under `logs/` and intentionally not committed.

## Summary

| Dataset | Setting | WER | Step 0 entropy | Final post-update entropy | Final update |
| --- | ---: | ---: | ---: | ---: | ---: |
| TED-LIUM test | `freq_mask` | 0.0579 | 0.1315 | 0.0710 | 400 |
| TED-LIUM test | `no_aug` | 0.0612 | 0.1931 | 0.0249 | 400 |
| Earnings22 test | `freq_mask` | 0.1501 | 0.2688 | 0.0673 | 1025 |
| Earnings22 test | `no_aug` | 0.1841 | 0.2658 | 0.0110 | 1025 |

The no-augmentation pseudo-label-only setting drives entropy substantially lower by the final update on both datasets, while the frequency-mask setting retains higher posterior entropy and also gives lower WER in these one-repeat test runs.

Validation:

```bash
cd lcasr
python3 results/entropy_ablation/aggregate_entropy.py
python3 results/entropy_ablation/plot_entropy.py --refresh
git diff --check
```
