#!/usr/bin/env bash
set -euo pipefail

# Earnings22 CTC self-training ablation: drop exactly one module from the
# one-epoch adaptation update, sweeping the requested learning rates.
#
# This script only runs when called explicitly. It is intended for PR review
# before launching on Stanage/local GPUs.

DATASET=${DATASET:-earnings22}
SPLIT=${SPLIT:-test}
EPOCH=${EPOCH:-1}
SEQ=${SEQ:-16384}
OVERLAP=${OVERLAP:-14336}
REPEATS=${REPEATS:-1}
GPU=${GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
LRS_STR=${LRS:-"9e-6 9e-5 9e-4"}
# Default checkpoint has 6 encoder layers (0..5). Override if using a different model.
LAYERS_STR=${LAYERS:-"0 1 2 3 4 5"}
INCLUDE_SUBSAMPLING=${INCLUDE_SUBSAMPLING:-1}
INCLUDE_CTC_DECODER=${INCLUDE_CTC_DECODER:-1}
RESULTS_DIR=${RESULTS_DIR:-"./results/ctc_self_training_layer_drop_lr_sweep"}
LOG_DIR="${RESULTS_DIR}/logs"

read -r -a LRS <<< "$LRS_STR"
read -r -a LAYERS <<< "$LAYERS_STR"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

run_item() {
    local lr="$1"
    local module_tag="$2"
    shift 2

    local lr_tag
    lr_tag=$(echo "$lr" | sed 's/-/m/g; s/+//g; s/\./p/g')
    local save_path="${RESULTS_DIR}/${DATASET}-${SPLIT}-epoch-${EPOCH}-lr-${lr_tag}-drop-${module_tag}.pkl"
    local log_path="${LOG_DIR}/${DATASET}-${SPLIT}-epoch-${EPOCH}-lr-${lr_tag}-drop-${module_tag}.log"

    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting CTC self-training layer-drop item lr=${lr} drop=${module_tag}" | tee -a "$log_path"
    echo "save_path=$save_path" | tee -a "$log_path"
    echo "log_path=$log_path" | tee -a "$log_path"

    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_dynamic_eval_full.py \
        -dfa \
        -epochs "$EPOCH" \
        -seq "$SEQ" \
        -o "$OVERLAP" \
        -split "$SPLIT" \
        -d "$DATASET" \
        -r "$REPEATS" \
        "$@" \
        -kwargs optim_lr="$lr" spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 \
        -s "$save_path" 2>&1 | tee -a "$log_path"
}

for lr in "${LRS[@]}"
do
    run_item "$lr" "none"

    if [ "$INCLUDE_SUBSAMPLING" = "1" ]; then
        run_item "$lr" "subsampling" --freeze_subsampling
    fi

    if [ "$INCLUDE_CTC_DECODER" = "1" ]; then
        run_item "$lr" "ctc-decoder" --freeze_ctc_decoder
    fi

    for layer in "${LAYERS[@]}"
    do
        run_item "$lr" "layer-${layer}" --freeze_layer "$layer"
    done
done
