#!/usr/bin/env bash
set -euo pipefail

# Additional Earnings22 CTC self-training ablations.
# All selected sweep families run sequentially on one runtime-selected GPU.
#
# Runtime controls:
#   GPU=2 bash launch_scripts/tune_ctc_self_training_extra_ablation_sweeps.sh
#   FAMILIES="train_only progressive_top progressive_bottom layer_type"
#   LRS="9e-6 9e-5 9e-4"                            # default
#   DRY_RUN=1                                       # print commands only

DATASET=${DATASET:-earnings22}
SPLIT=${SPLIT:-test}
EPOCH=${EPOCH:-1}
SEQ=${SEQ:-16384}
OVERLAP=${OVERLAP:-14336}
REPEATS=${REPEATS:-1}
GPU=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
LRS_STR=${LRS:-"9e-6 9e-5 9e-4"}
# Default checkpoint has 6 encoder layers (0..5). Override if using a different model.
LAYERS_STR=${LAYERS:-"0 1 2 3 4 5"}
FAMILIES_STR=${FAMILIES:-"train_only progressive_top layer_type"}
RESULTS_ROOT=${RESULTS_ROOT:-"./results/ctc_self_training_extra_ablation_sweeps"}
LOG_ROOT="${RESULTS_ROOT}/logs"
DRY_RUN=${DRY_RUN:-0}

read -r -a LRS <<< "$LRS_STR"
read -r -a LAYERS <<< "$LAYERS_STR"
read -r -a FAMILIES <<< "$FAMILIES_STR"

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT"

run_item() {
    local family="$1"
    local lr="$2"
    local tag="$3"
    shift 3

    local lr_tag
    lr_tag=$(echo "$lr" | sed 's/-/m/g; s/+//g; s/\./p/g')
    local family_dir="${RESULTS_ROOT}/${family}"
    local log_dir="${LOG_ROOT}/${family}"
    mkdir -p "$family_dir" "$log_dir"

    local save_path="${family_dir}/${DATASET}-${SPLIT}-epoch-${EPOCH}-lr-${lr_tag}-${tag}.pkl"
    local log_path="${log_dir}/${DATASET}-${SPLIT}-epoch-${EPOCH}-lr-${lr_tag}-${tag}.log"

    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${family} item lr=${lr} tag=${tag}" | tee -a "$log_path"
    echo "gpu=$GPU" | tee -a "$log_path"
    echo "save_path=$save_path" | tee -a "$log_path"
    echo "log_path=$log_path" | tee -a "$log_path"

    local -a cmd=(
        "$PYTHON_BIN" run_dynamic_eval_full.py
        -dfa \
        -epochs "$EPOCH"
        -seq "$SEQ"
        -o "$OVERLAP"
        -split "$SPLIT"
        -d "$DATASET"
        -r "$REPEATS"
    )
    cmd+=("$@")
    cmd+=(
        -kwargs optim_lr="$lr" spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0
        -s "$save_path"
    )

    if [ "$DRY_RUN" = "1" ]; then
        printf 'DRY_RUN CUDA_VISIBLE_DEVICES=%q' "$GPU" | tee -a "$log_path"
        printf ' %q' "${cmd[@]}" | tee -a "$log_path"
        printf '\n' | tee -a "$log_path"
        return
    fi

    CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" 2>&1 | tee -a "$log_path"
}

run_train_only_family() {
    for lr in "${LRS[@]}"
    do
        run_item "train_only" "$lr" "train-all"
        run_item "train_only" "$lr" "train-subsampling-only" --train_subsampling_only
        run_item "train_only" "$lr" "train-ctc-decoder-only" --train_only_ctc_decoder
        for layer in "${LAYERS[@]}"
        do
            run_item "train_only" "$lr" "train-layer-${layer}-only" --train_only_layer "$layer"
        done
    done
}

run_progressive_top_family() {
    for lr in "${LRS[@]}"
    do
        run_item "progressive_top" "$lr" "train-all"
        run_item "progressive_top" "$lr" "freeze-subsampling" --freeze_subsampling
        for layer in "${LAYERS[@]}"
        do
            run_item "progressive_top" "$lr" "freeze-subsampling-through-layer-${layer}" --freeze_subsampling --freeze_layers_through "$layer"
        done
    done
}

run_progressive_bottom_family() {
    for lr in "${LRS[@]}"
    do
        run_item "progressive_bottom" "$lr" "train-subsampling-only" --train_subsampling_only
        for layer in "${LAYERS[@]}"
        do
            run_item "progressive_bottom" "$lr" "train-subsampling-through-layer-${layer}" --train_layers_through "$layer"
        done
    done
}

run_layer_type_family() {
    for lr in "${LRS[@]}"
    do
        run_item "layer_type" "$lr" "train-attention-only" --train_only_layer_type attention
        run_item "layer_type" "$lr" "train-feed-forward-only" --train_only_layer_type feed_forward
        run_item "layer_type" "$lr" "train-convolution-only" --train_only_layer_type convolution
    done
}

for family in "${FAMILIES[@]}"
do
    case "$family" in
        train_only)
            run_train_only_family
            ;;
        progressive_top)
            run_progressive_top_family
            ;;
        progressive_bottom)
            run_progressive_bottom_family
            ;;
        layer_type)
            run_layer_type_family
            ;;
        *)
            echo "Unknown family: $family" >&2
            exit 1
            ;;
    esac
done
