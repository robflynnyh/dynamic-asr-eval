#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-earnings22}
SPLIT=${SPLIT:-test}
EPOCH=${EPOCH:-1}
SEQ=${SEQ:-16384}
OVERLAP=${OVERLAP:-14336}
REPEATS=${REPEATS:-3}
GPU=${GPU:-2}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
LRS_STR=${LRS:-"9e-7 9e-6 9e-5"}
read -r -a LRS <<< "$LRS_STR"
RESULTS_DIR="./results/dynamic_eval_lr_sweep"
LOG_DIR="${RESULTS_DIR}/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

for lr in "${LRS[@]}"
do
    lr_tag=$(echo "$lr" | sed 's/-/m/g; s/+//g; s/\./p/g')
    SAVE_PATH="${RESULTS_DIR}/${DATASET}-${SPLIT}-epoch-${EPOCH}-lr-${lr_tag}.pkl"
    LOG_PATH="${LOG_DIR}/${DATASET}-${SPLIT}-epoch-${EPOCH}-lr-${lr_tag}.log"

    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting dynamic eval sweep item lr=${lr}" | tee -a "$LOG_PATH"
    echo "save_path=$SAVE_PATH" | tee -a "$LOG_PATH"
    echo "log_path=$LOG_PATH" | tee -a "$LOG_PATH"

    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run.py \
        -dfa \
        -epochs "$EPOCH" \
        -seq "$SEQ" \
        -o "$OVERLAP" \
        -split "$SPLIT" \
        -d "$DATASET" \
        -r "$REPEATS" \
        -kwargs optim_lr="$lr" spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 \
        -s "$SAVE_PATH" 2>&1 | tee -a "$LOG_PATH"
done
