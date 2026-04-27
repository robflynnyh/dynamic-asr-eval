#!/usr/bin/env bash
set -euo pipefail

EPOCHS_STR=${EPOCHS:-"1 5"}
read -r -a EPOCHS <<< "$EPOCHS_STR"
DATASET=${DATASET:-earnings22}
SPLIT=${SPLIT:-test}
SEQ=${SEQ:-16384}
OVERLAP=${OVERLAP:-14336}
ADAPT_OVERLAP=${ADAPT_OVERLAP:-14336}
REPEATS=${REPEATS:-3}
SPLIT_SEED=${SPLIT_SEED:-0}
SHUFFLE_SPLITS=${SHUFFLE_SPLITS:-0}
GPU=${GPU:-3}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
RESULTS_DIR="./results/half_concat_eval"
LOG_DIR="${RESULTS_DIR}/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

EXTRA_ARGS=()
if [[ "$SHUFFLE_SPLITS" == "1" ]]; then
    EXTRA_ARGS+=(--shuffle_splits)
fi

for epoch in "${EPOCHS[@]}"
do
    SAVE_PATH="${RESULTS_DIR}/${DATASET}-${SPLIT}-half-concat-epoch-${epoch}.pkl"
    LOG_PATH="${LOG_DIR}/${DATASET}-${SPLIT}-half-concat-epoch-${epoch}.log"

    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting half-concat eval epoch=${epoch}" | tee -a "$LOG_PATH"
    echo "save_path=$SAVE_PATH" | tee -a "$LOG_PATH"
    echo "log_path=$LOG_PATH" | tee -a "$LOG_PATH"

    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_half_concat_eval.py \
        -dfa \
        -epochs "$epoch" \
        -seq "$SEQ" \
        -o "$OVERLAP" \
        -ao "$ADAPT_OVERLAP" \
        -split "$SPLIT" \
        -d "$DATASET" \
        -r "$REPEATS" \
        --split_seed "$SPLIT_SEED" \
        "${EXTRA_ARGS[@]}" \
        -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 \
        -s "$SAVE_PATH" 2>&1 | tee -a "$LOG_PATH"
done
