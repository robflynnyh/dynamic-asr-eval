#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-earnings22}
SPLIT=${SPLIT:-test}
EPOCH=${EPOCH:-1}
SEQ=${SEQ:-16384}
OVERLAP=${OVERLAP:-14336}
REPEATS=${REPEATS:-1}
GPU=${GPU:-2}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
FREEZE_SUBSAMPLING=${FREEZE_SUBSAMPLING:-0}
FREEZE_ALL_BUT_LAST_BLOCK_AND_HEAD=${FREEZE_ALL_BUT_LAST_BLOCK_AND_HEAD:-0}
TRAIN_SUBSAMPLING_ONLY=${TRAIN_SUBSAMPLING_ONLY:-0}
LRS_STR=${LRS:-"9e-5"}
read -r -a LRS <<< "$LRS_STR"
RESULTS_SUFFIX=""
if [ "$FREEZE_SUBSAMPLING" = "1" ]; then
    RESULTS_SUFFIX="_freeze_subsampling"
fi
if [ "$FREEZE_ALL_BUT_LAST_BLOCK_AND_HEAD" = "1" ]; then
    RESULTS_SUFFIX="${RESULTS_SUFFIX}_freeze_last_block_and_head"
fi
if [ "$TRAIN_SUBSAMPLING_ONLY" = "1" ]; then
    RESULTS_SUFFIX="${RESULTS_SUFFIX}_train_subsampling_only"
fi
RESULTS_DIR="./results/dynamic_eval_lr_sweep${RESULTS_SUFFIX}"
LOG_DIR="${RESULTS_DIR}/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

for lr in "${LRS[@]}"
do
    lr_tag=$(echo "$lr" | sed 's/-/m/g; s/+//g; s/\./p/g')
    suffix=""
    if [ "$FREEZE_SUBSAMPLING" = "1" ]; then
        suffix="-freeze-subsampling"
    fi
    if [ "$FREEZE_ALL_BUT_LAST_BLOCK_AND_HEAD" = "1" ]; then
        suffix="${suffix}-freeze-last-block-and-head"
    fi
    if [ "$TRAIN_SUBSAMPLING_ONLY" = "1" ]; then
        suffix="${suffix}-train-subsampling-only"
    fi
    SAVE_PATH="${RESULTS_DIR}/${DATASET}-${SPLIT}-epoch-${EPOCH}-lr-${lr_tag}${suffix}.pkl"
    LOG_PATH="${LOG_DIR}/${DATASET}-${SPLIT}-epoch-${EPOCH}-lr-${lr_tag}${suffix}.log"

    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting dynamic eval lr sweep item lr=${lr}" | tee -a "$LOG_PATH"
    echo "freeze_subsampling=$FREEZE_SUBSAMPLING" | tee -a "$LOG_PATH"
    echo "freeze_all_but_last_block_and_head=$FREEZE_ALL_BUT_LAST_BLOCK_AND_HEAD" | tee -a "$LOG_PATH"
    echo "train_subsampling_only=$TRAIN_SUBSAMPLING_ONLY" | tee -a "$LOG_PATH"
    echo "save_path=$SAVE_PATH" | tee -a "$LOG_PATH"
    echo "log_path=$LOG_PATH" | tee -a "$LOG_PATH"

    extra_args=()
    if [ "$FREEZE_SUBSAMPLING" = "1" ]; then
        extra_args+=(--freeze_subsampling)
    fi
    if [ "$FREEZE_ALL_BUT_LAST_BLOCK_AND_HEAD" = "1" ]; then
        extra_args+=(--freeze_all_but_last_block_and_head)
    fi
    if [ "$TRAIN_SUBSAMPLING_ONLY" = "1" ]; then
        extra_args+=(--train_subsampling_only)
    fi

    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_dynamic_eval_full.py \
        -dfa \
        -epochs "$EPOCH" \
        -seq "$SEQ" \
        -o "$OVERLAP" \
        -split "$SPLIT" \
        -d "$DATASET" \
        -r "$REPEATS" \
        "${extra_args[@]}" \
        -kwargs optim_lr="$lr" spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 \
        -s "$SAVE_PATH" 2>&1 | tee -a "$LOG_PATH"
done
