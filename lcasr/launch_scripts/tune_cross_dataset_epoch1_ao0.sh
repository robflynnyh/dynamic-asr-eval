#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3.10}
EPOCH=1
LOG_DIR="./results/crossdataset/logs"
SAVE_PATH="./results/crossdataset/earnings22_tedlium-epoch-${EPOCH}-ao0-test.pkl"
LOG_PATH="${LOG_DIR}/earnings22_tedlium-epoch-${EPOCH}-ao0-test.log"

mkdir -p "$LOG_DIR"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting cross-dataset eval" | tee -a "$LOG_PATH"
echo "save_path=$SAVE_PATH" | tee -a "$LOG_PATH"
echo "log_path=$LOG_PATH" | tee -a "$LOG_PATH"

CUDA_VISIBLE_DEVICES="3" "$PYTHON_BIN" run_cross_dataset_eval.py \
  -dfa \
  -epochs "$EPOCH" \
  -seq 16384 \
  -o 14336 \
  -ao 0 \
  -split test \
  -d earnings22 \
  -d2 tedlium \
  -r 3 \
  -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 \
  -s "$SAVE_PATH" 2>&1 | tee -a "$LOG_PATH"
