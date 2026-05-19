# OpenClaw Launch Note

Launch the TEDLIUM-dev teacher-KL epoch-relabel stability sweep from the repo
root:

```bash
cd /exp/exp4/acp21rjf/dynamic-asr-eval/lcasr
GPU=<free_gpu>
screen -L -Logfile results/enc_dec/enc_dec_v2/enc_dec_teacher_epoch_relabel/logs/screen_tedlium_dev_epoch_relabel_kl_gpu${GPU}.log -dmS tedlium_dev_epoch_relabel_kl_gpu${GPU} bash -lc "GPU=${GPU} bash launch_scripts/tune_enc_dec_teacher_epoch_relabel_tedlium_dev.sh"
```

Choose `<free_gpu>` from `nvidia-smi` by checking compute-process occupancy, not
just low memory. For example, if GPU 3 is free:

```bash
GPU=3
screen -L -Logfile results/enc_dec/enc_dec_v2/enc_dec_teacher_epoch_relabel/logs/screen_tedlium_dev_epoch_relabel_kl_gpu${GPU}.log -dmS tedlium_dev_epoch_relabel_kl_gpu${GPU} bash -lc "GPU=${GPU} bash launch_scripts/tune_enc_dec_teacher_epoch_relabel_tedlium_dev.sh"
```

The launcher runs:

- dataset/split: `tedlium/dev`
- training: `teacher_kl`
- timing: `--teacher_epoch_relabel`
- decode: `beam=5`, `enc_dec_length_penalty=0.5`
- repeats: `1`
- LRs: `3e-7 1e-7 1e-8`
- KL temps: `1.0 0.7 0.5`
- filters: `relaxed ctc strict_ctc`
- augmentations: `no_aug freq2_width16_time0 freq3_width24_time0 freq6_width34_time0 freq8_width48_time0 freq2_width16_time1`

Expected size: `162` adaptation/eval settings plus one no-adapt baseline.

Monitor:

```bash
screen -ls
tail -f results/enc_dec/enc_dec_v2/enc_dec_teacher_epoch_relabel/logs/screen_tedlium_dev_epoch_relabel_kl_gpu${GPU}.log
```

Aggregate after results finish:

```bash
python results/enc_dec/enc_dec_v2/enc_dec_teacher_epoch_relabel/aggregate.py --csv results/enc_dec/enc_dec_v2/enc_dec_teacher_epoch_relabel/summary.csv
```
