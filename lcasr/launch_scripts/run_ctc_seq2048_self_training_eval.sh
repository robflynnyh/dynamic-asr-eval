#!/usr/bin/env bash
set -euo pipefail

# ROB-56: CTC self-training eval with the 1-epoch 2048-context checkpoint.
# Run from lcasr/ directly. When launched through with-gpu, inherit its
# CUDA_VISIBLE_DEVICES assignment; set GPU manually only for local one-off runs.

DATASETS_STR=${DATASETS:-"earnings22 tedlium chime6 rev16"}
EPOCHS_STR=${EPOCHS:-"1 5"}
SPLIT=${SPLIT:-test}
SEQ=${SEQ:-2048}
OVERLAP=${OVERLAP:-1792}
REPEATS=${REPEATS:-1}
LR=${LR:-9e-5}
CHECKPOINT=${CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt"}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
RESULTS_DIR=${RESULTS_DIR:-"./results/ctc_seq2048_self_training_eval"}
LOG_DIR="${RESULTS_DIR}/logs"
DRY_RUN=${DRY_RUN:-0}

read -r -a DATASETS <<< "$DATASETS_STR"
read -r -a EPOCHS <<< "$EPOCHS_STR"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

lr_tag() {
    echo "$1" | sed 's/-/m/g; s/+//g; s/\./p/g'
}

run_eval() {
    local dataset="$1"
    local epoch="$2"
    local lr_slug
    lr_slug=$(lr_tag "$LR")

    local base_name="${dataset}-${SPLIT}-ctc-seq${SEQ}-overlap${OVERLAP}-epoch-${epoch}-lr-${lr_slug}"
    local save_path="${RESULTS_DIR}/${base_name}.pkl"
    local log_path="${LOG_DIR}/${base_name}.log"

    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${base_name}" | tee -a "$log_path"
    echo "checkpoint=${CHECKPOINT}" | tee -a "$log_path"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$log_path"
    echo "manual_gpu=${GPU:-unset}" | tee -a "$log_path"
    echo "save_path=${save_path}" | tee -a "$log_path"

    local cmd=(
        "$PYTHON_BIN" run_dynamic_eval_full.py
        -c "$CHECKPOINT"
        -dfa
        -epochs "$epoch"
        -seq "$SEQ"
        -o "$OVERLAP"
        -split "$SPLIT"
        -d "$dataset"
        -r "$REPEATS"
        -kwargs
        optim_lr="$LR"
        spec_augment_n_freq_masks=6
        spec_augment_freq_mask_param=34
        spec_augment_n_time_masks=0
        -s "$save_path"
    )

    if [ "$DRY_RUN" = "1" ]; then
        printf 'DRY_RUN'
        if [ -n "${GPU:-}" ]; then
            printf ' CUDA_VISIBLE_DEVICES=%q' "$GPU"
        fi
        printf ' %q' "${cmd[@]}"
        printf '\n' | tee -a "$log_path"
        return
    fi

    if [ -n "${GPU:-}" ]; then
        CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" 2>&1 | tee -a "$log_path"
    else
        "${cmd[@]}" 2>&1 | tee -a "$log_path"
    fi
}

for dataset in "${DATASETS[@]}"; do
    for epoch in "${EPOCHS[@]}"; do
        run_eval "$dataset" "$epoch"
    done
done
