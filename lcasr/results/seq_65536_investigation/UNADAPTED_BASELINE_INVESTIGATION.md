# ROB-67 65536 Unadapted Baseline Investigation

This checks whether the 65536-context before-adaptation CTC rows look anomalous because of a reporting/configuration mismatch or because the checkpoint genuinely behaves similarly to shorter-context baselines before self-training.

## Conclusion

- No aggregation or launcher-contract mismatch was found in the committed 65536 unadapted artifacts.
- Each 65536 PKL reports `epochs=0`, greedy CTC (`beamsearch=False`), no AWMC/consistency path, `split=test`, `seq_len=65536`, `overlap=57344`, one repeat, and no `max_records` cap.
- Compared with the exact committed 2048 no-adapt baseline, 65536 is better on TEDLIUM and Earnings22, effectively tied on Rev16, and worse on CHiME-6.
- CHiME-6 is the main suspicious row: the 65536 model has a higher deletion rate and a lower output/reference word ratio than the 2048 baseline.
- The committed tree does not contain an all-dataset exact 16384 no-adapt CTC baseline analogous to ROB-66/ROB-67, so the 2.7 minute comparison cannot be fully validated from committed exact-match artifacts alone.
- The 65536 and 2048 baselines use the same reference record sets for all four datasets.
- CHiME-6 and Earnings22 appear in a different record order between the 2048 and 65536 runs; this changes concatenated hashes but not each run's paired WER calculation.

## 65536 vs 2048 No-Adapt Baseline

| Dataset | 65536 WER | 2048 WER | Delta | 65536 Del | 2048 Del | 65536 Out/Ref | 2048 Out/Ref | Reference set matches | Reference order matches |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| chime6 | 86.76% | 85.27% | +1.49 pp | 84.51% | 82.62% | 0.156 | 0.175 | True | False |
| earnings22 | 18.32% | 19.54% | -1.22 pp | 5.17% | 5.09% | 0.970 | 0.975 | True | False |
| rev16 | 15.21% | 15.27% | -0.06 pp | 7.13% | 6.87% | 0.946 | 0.949 | True | True |
| tedlium | 6.17% | 6.54% | -0.38 pp | 2.05% | 2.00% | 0.987 | 0.988 | True | True |

## Run Metadata Checks

| Dataset | 65536 checkpoint | 65536 cfg chunk | 65536 seed | 2048 checkpoint | 2048 cfg chunk | 2048 seed |
|---|---|---:|---:|---|---:|---:|
| chime6 | `/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_65536_rp_1/step_105360.pt` | 512 | 4282 | `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt` | 2048 | 9643 |
| earnings22 | `/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_65536_rp_1/step_105360.pt` | 512 | 4282 | `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt` | 2048 | 9643 |
| rev16 | `/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_65536_rp_1/step_105360.pt` | 512 | 4282 | `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt` | 2048 | 9643 |
| tedlium | `/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_65536_rp_1/step_105360.pt` | 512 | 4282 | `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt` | 2048 | 9643 |

## Legacy 16384 Context Note

A legacy Earnings22 `epochs=0` artifact exists at `lcasr/results/paper/per_epoch_eval/epoch-0-earnings22-test_1.pkl` with WER 18.32%, `seq_len=16384`, `overlap=14336`, and checkpoint `/store/store5/data/acp21rjf_checkpoints/lcasr/lcasr-160-rb_e1/step_105360.pt`.
It matches the current 65536 Earnings22 WER to the shown precision, but it is not the same artifact: checkpoint paths differ, checkpoint config seeds differ, output hashes differ (`238cbdb23f5a` vs `fd9c7bc98269`), and the sorted reference-set hash comparison is `True`.

## Reproduce

```bash
python3 lcasr/results/seq_65536_investigation/aggregate_unadapted.py
python3 lcasr/results/seq_65536_investigation/investigate_unadapted_baseline.py
```

The script writes `unadapted_baseline/investigation.csv` and this report.
