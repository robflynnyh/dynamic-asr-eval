#!/usr/bin/env bash
set -euo pipefail

EPOCHS_STR=${EPOCHS:-"1 3 5 10"}
read -r -a EPOCHS <<< "$EPOCHS_STR"
LRS_STR=${LRS:-"9e-6 9e-5"}
read -r -a LRS <<< "$LRS_STR"
DATASET=${DATASET:-earnings22}
SPLIT=${SPLIT:-test}
SEQ=${SEQ:-16384}
OVERLAP=${OVERLAP:-14336}
ADAPT_OVERLAP=${ADAPT_OVERLAP:-14336}
REPEATS=${REPEATS:-3}
GPU=${GPU:-3}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
RESULTS_DIR="./results/whole_concat_eval"
LOG_DIR="${RESULTS_DIR}/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

for lr in "${LRS[@]}"
do
    LR_TAG=${lr//-/m}
    for epoch in "${EPOCHS[@]}"
    do
        SAVE_PATH="${RESULTS_DIR}/${DATASET}-${SPLIT}-whole-concat-epoch-${epoch}-lr-${LR_TAG}.pkl"
        LOG_PATH="${LOG_DIR}/${DATASET}-${SPLIT}-whole-concat-epoch-${epoch}-lr-${LR_TAG}.log"

        echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting whole-concat eval epoch=${epoch} lr=${lr}" | tee -a "$LOG_PATH"
        echo "save_path=$SAVE_PATH" | tee -a "$LOG_PATH"
        echo "log_path=$LOG_PATH" | tee -a "$LOG_PATH"

        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_whole_concat_eval.py \
            -dfa \
            -epochs "$epoch" \
            -seq "$SEQ" \
            -o "$OVERLAP" \
            -ao "$ADAPT_OVERLAP" \
            -split "$SPLIT" \
            -d "$DATASET" \
            -r "$REPEATS" \
            -kwargs optim_lr=${lr} spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 \
            -s "$SAVE_PATH" 2>&1 | tee -a "$LOG_PATH"
    done
done
