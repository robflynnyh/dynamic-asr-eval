# ROB-61 checkpoint benchmark

Normal encoder-decoder inference benchmark; no self-training or test-time adaptation.
Greedy uses the default autoregressive decode. Beam uses `beam_width=5` and `length_penalty=0.5`.

| Dataset | Decode | Checkpoint | WER | Delta vs old | Relative delta | Ins | Del | Sub |
|---|---|---|---:|---:|---:|---:|---:|---:|
| earnings22 | beam5_lp0p5 | old_seed | 0.25172 | +0.00000 | +0.00% | 0.04352 | 0.05518 | 0.15301 |
| earnings22 | beam5_lp0p5 | rl_step_10000 | 0.23369 | -0.01803 | -7.16% | 0.02931 | 0.05745 | 0.14693 |
| earnings22 | beam5_lp0p5 | rl_step_2000 | 0.23418 | -0.01754 | -6.97% | 0.02955 | 0.06000 | 0.14462 |
| earnings22 | beam5_lp0p5 | rl_step_20000 | 0.23070 | -0.02102 | -8.35% | 0.02906 | 0.05543 | 0.14621 |
| earnings22 | beam5_lp0p5 | rl_step_30000 | 0.23007 | -0.02165 | -8.60% | 0.02829 | 0.05666 | 0.14513 |
| earnings22 | greedy | old_seed | 0.26657 | +0.00000 | +0.00% | 0.04873 | 0.05431 | 0.16353 |
| earnings22 | greedy | rl_step_10000 | 0.25027 | -0.01630 | -6.11% | 0.03204 | 0.06013 | 0.15810 |
| earnings22 | greedy | rl_step_2000 | 0.25237 | -0.01419 | -5.32% | 0.03429 | 0.05808 | 0.16000 |
| earnings22 | greedy | rl_step_20000 | 0.24588 | -0.02069 | -7.76% | 0.03211 | 0.05745 | 0.15632 |
| earnings22 | greedy | rl_step_30000 | 0.24543 | -0.02114 | -7.93% | 0.03037 | 0.05966 | 0.15540 |
| tedlium | beam5_lp0p5 | old_seed | 0.08896 | +0.00000 | +0.00% | 0.01216 | 0.03349 | 0.04331 |
| tedlium | beam5_lp0p5 | rl_step_10000 | 0.08456 | -0.00439 | -4.94% | 0.00815 | 0.03317 | 0.04324 |
| tedlium | beam5_lp0p5 | rl_step_2000 | 0.08623 | -0.00273 | -3.07% | 0.00829 | 0.03480 | 0.04313 |
| tedlium | beam5_lp0p5 | rl_step_20000 | 0.08425 | -0.00471 | -5.30% | 0.00893 | 0.03176 | 0.04356 |
| tedlium | beam5_lp0p5 | rl_step_30000 | 0.08386 | -0.00510 | -5.74% | 0.00886 | 0.03243 | 0.04257 |
| tedlium | greedy | old_seed | 0.09516 | +0.00000 | +0.00% | 0.01354 | 0.03569 | 0.04593 |
| tedlium | greedy | rl_step_10000 | 0.09498 | -0.00018 | -0.19% | 0.01481 | 0.03257 | 0.04760 |
| tedlium | greedy | rl_step_2000 | 0.09197 | -0.00319 | -3.35% | 0.01014 | 0.03456 | 0.04728 |
| tedlium | greedy | rl_step_20000 | 0.09328 | -0.00188 | -1.97% | 0.01503 | 0.03115 | 0.04710 |
| tedlium | greedy | rl_step_30000 | 0.08900 | -0.00617 | -6.48% | 0.01007 | 0.03169 | 0.04724 |
