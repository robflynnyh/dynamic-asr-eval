#!/usr/bin/env bash
set -euo pipefail

# Fill missing repeat files for the CTC self-training extra ablations.
#
# Defaults:
#   1. Run repeats 2 and 3 for the existing extra-ablation grids:
#        - train_only, progressive_top, layer_type
#        - Earnings22 at 9e-6/9e-5/9e-4
#        - TEDLIUM at 9e-5
#        - layer_drop_lr_sweep on Earnings22 at 9e-6/9e-5/9e-4
#   2. Run repeats 1, 2, and 3 for the missing TEDLIUM layer-drop ablation at 9e-5.
#
# Example:
#   GPU=1 bash launch_scripts/run_ctc_self_training_extra_ablation_missing_repeats.sh
#
# Useful overrides:
#   EXTRA_REPEATS="2 3"
#   LAYER_DROP_TEDLIUM_REPEATS="1 2 3"
#   EXTRA_ABLATION_FAMILIES="train_only progressive_top layer_type"
#   EXTRA_ABLATION_EARNINGS22_LRS="9e-6 9e-5 9e-4"
#   EXTRA_ABLATION_TEDLIUM_LRS="9e-5"
#   LAYER_DROP_EARNINGS22_LRS="9e-6 9e-5 9e-4"
#   LAYER_DROP_TEDLIUM_LRS="9e-5"

DATASET_SPLIT=${DATASET_SPLIT:-test}
EPOCH=${EPOCH:-1}
SEQ=${SEQ:-16384}
OVERLAP=${OVERLAP:-14336}
GPU=${GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
RESULTS_ROOT=${RESULTS_ROOT:-"./results/ctc_self_training_extra_ablation_sweeps"}
LOG_ROOT="${RESULTS_ROOT}/logs"
LAYERS_STR=${LAYERS:-"0 1 2 3 4 5"}
EXTRA_REPEATS_STR=${EXTRA_REPEATS:-"2 3"}
LAYER_DROP_TEDLIUM_REPEATS_STR=${LAYER_DROP_TEDLIUM_REPEATS:-"1 2 3"}
EXTRA_ABLATION_FAMILIES_STR=${EXTRA_ABLATION_FAMILIES:-"train_only progressive_top layer_type"}
EXTRA_ABLATION_EARNINGS22_LRS_STR=${EXTRA_ABLATION_EARNINGS22_LRS:-"9e-6 9e-5 9e-4"}
EXTRA_ABLATION_TEDLIUM_LRS_STR=${EXTRA_ABLATION_TEDLIUM_LRS:-"9e-5"}
LAYER_DROP_EARNINGS22_LRS_STR=${LAYER_DROP_EARNINGS22_LRS:-"9e-6 9e-5 9e-4"}
LAYER_DROP_TEDLIUM_LRS_STR=${LAYER_DROP_TEDLIUM_LRS:-"9e-5"}
SKIP_EXISTING=${SKIP_EXISTING:-1}
DRY_RUN=${DRY_RUN:-0}

read -r -a LAYERS <<< "$LAYERS_STR"
read -r -a EXTRA_REPEATS <<< "$EXTRA_REPEATS_STR"
read -r -a LAYER_DROP_TEDLIUM_REPEATS <<< "$LAYER_DROP_TEDLIUM_REPEATS_STR"
read -r -a EXTRA_ABLATION_FAMILIES <<< "$EXTRA_ABLATION_FAMILIES_STR"
read -r -a EXTRA_ABLATION_EARNINGS22_LRS <<< "$EXTRA_ABLATION_EARNINGS22_LRS_STR"
read -r -a EXTRA_ABLATION_TEDLIUM_LRS <<< "$EXTRA_ABLATION_TEDLIUM_LRS_STR"
read -r -a LAYER_DROP_EARNINGS22_LRS <<< "$LAYER_DROP_EARNINGS22_LRS_STR"
read -r -a LAYER_DROP_TEDLIUM_LRS <<< "$LAYER_DROP_TEDLIUM_LRS_STR"

lr_tag() {
    echo "$1" | sed 's/-/m/g; s/+//g; s/\./p/g'
}

patch_repeat_metadata() {
    local path="$1"
    local base_save_path="$2"
    local repeat="$3"
    local total_repeats="$4"
    "$PYTHON_BIN" - "$path" "$base_save_path" "$repeat" "$total_repeats" <<'PY'
import pickle
import sys
from pathlib import Path

path = Path(sys.argv[1])
base_save_path = sys.argv[2]
repeat = sys.argv[3]
total_repeats = sys.argv[4]

with path.open("rb") as f:
    data = pickle.load(f)
data["repeat"] = f"{repeat}/{total_repeats}"
if isinstance(data.get("args_dict"), dict):
    data["args_dict"]["repeats"] = int(total_repeats)
    data["args_dict"]["save_path"] = base_save_path
with path.open("wb") as f:
    pickle.dump(data, f)
PY
}

run_one_repeat() {
    local dataset="$1"
    local group="$2"
    local lr="$3"
    local tag="$4"
    local repeat="$5"
    local total_repeats="$6"
    shift 6

    local lr_slug
    lr_slug=$(lr_tag "$lr")
    local group_dir="${RESULTS_ROOT}/${group}"
    local log_dir="${LOG_ROOT}/${group}"
    mkdir -p "$group_dir" "$log_dir"

    local base_path="${group_dir}/${dataset}-${DATASET_SPLIT}-epoch-${EPOCH}-lr-${lr_slug}-${tag}.pkl"
    local target_path="${base_path%.pkl}_${repeat}.pkl"
    local tmp_base="${group_dir}/${dataset}-${DATASET_SPLIT}-epoch-${EPOCH}-lr-${lr_slug}-${tag}.repeat-${repeat}.tmp.pkl"
    local tmp_output="${tmp_base%.pkl}_1.pkl"
    local log_path="${log_dir}/${dataset}-${DATASET_SPLIT}-epoch-${EPOCH}-lr-${lr_slug}-${tag}_repeat-${repeat}.log"

    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$target_path" ]; then
        echo "Skipping existing $target_path"
        return
    fi

    rm -f "$tmp_output"
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${group} dataset=${dataset} lr=${lr} tag=${tag} repeat=${repeat}/${total_repeats}" | tee -a "$log_path"
    echo "gpu=$GPU" | tee -a "$log_path"
    echo "target_path=$target_path" | tee -a "$log_path"

    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY_RUN CUDA_VISIBLE_DEVICES=$GPU $PYTHON_BIN run_dynamic_eval_full.py -dfa -epochs $EPOCH -seq $SEQ -o $OVERLAP -split $DATASET_SPLIT -d $dataset -r 1 $* -kwargs optim_lr=$lr spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 -s $tmp_base" | tee -a "$log_path"
        return
    fi

    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_dynamic_eval_full.py \
        -dfa \
        -epochs "$EPOCH" \
        -seq "$SEQ" \
        -o "$OVERLAP" \
        -split "$DATASET_SPLIT" \
        -d "$dataset" \
        -r 1 \
        "$@" \
        -kwargs optim_lr="$lr" spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 \
        -s "$tmp_base" 2>&1 | tee -a "$log_path"

    patch_repeat_metadata "$tmp_output" "$base_path" "$repeat" "$total_repeats"
    mv "$tmp_output" "$target_path"
    echo "Saved repeat ${repeat}/${total_repeats} to $target_path" | tee -a "$log_path"
}

run_train_only_family() {
    local dataset="$1"
    local lr="$2"
    local repeat="$3"
    local total_repeats="$4"
    run_one_repeat "$dataset" "train_only" "$lr" "train-all" "$repeat" "$total_repeats"
    run_one_repeat "$dataset" "train_only" "$lr" "train-subsampling-only" "$repeat" "$total_repeats" --train_subsampling_only
    run_one_repeat "$dataset" "train_only" "$lr" "train-ctc-decoder-only" "$repeat" "$total_repeats" --train_only_ctc_decoder
    for layer in "${LAYERS[@]}"; do
        run_one_repeat "$dataset" "train_only" "$lr" "train-layer-${layer}-only" "$repeat" "$total_repeats" --train_only_layer "$layer"
    done
}

run_progressive_top_family() {
    local dataset="$1"
    local lr="$2"
    local repeat="$3"
    local total_repeats="$4"
    run_one_repeat "$dataset" "progressive_top" "$lr" "train-all" "$repeat" "$total_repeats"
    run_one_repeat "$dataset" "progressive_top" "$lr" "freeze-subsampling" "$repeat" "$total_repeats" --freeze_subsampling
    for layer in "${LAYERS[@]}"; do
        run_one_repeat "$dataset" "progressive_top" "$lr" "freeze-subsampling-through-layer-${layer}" "$repeat" "$total_repeats" --freeze_subsampling --freeze_layers_through "$layer"
    done
}

run_layer_type_family() {
    local dataset="$1"
    local lr="$2"
    local repeat="$3"
    local total_repeats="$4"
    run_one_repeat "$dataset" "layer_type" "$lr" "train-attention-only" "$repeat" "$total_repeats" --train_only_layer_type attention
    run_one_repeat "$dataset" "layer_type" "$lr" "train-feed-forward-only" "$repeat" "$total_repeats" --train_only_layer_type feed_forward
    run_one_repeat "$dataset" "layer_type" "$lr" "train-convolution-only" "$repeat" "$total_repeats" --train_only_layer_type convolution
}

run_extra_ablation_family() {
    local family="$1"
    local dataset="$2"
    local lr="$3"
    local repeat="$4"
    local total_repeats="$5"
    case "$family" in
        train_only) run_train_only_family "$dataset" "$lr" "$repeat" "$total_repeats" ;;
        progressive_top) run_progressive_top_family "$dataset" "$lr" "$repeat" "$total_repeats" ;;
        layer_type) run_layer_type_family "$dataset" "$lr" "$repeat" "$total_repeats" ;;
        *) echo "Unknown extra ablation family: $family" >&2; exit 1 ;;
    esac
}

run_layer_drop_family() {
    local dataset="$1"
    local lr="$2"
    local repeat="$3"
    local total_repeats="$4"
    run_one_repeat "$dataset" "layer_drop_lr_sweep" "$lr" "drop-none" "$repeat" "$total_repeats"
    run_one_repeat "$dataset" "layer_drop_lr_sweep" "$lr" "drop-subsampling" "$repeat" "$total_repeats" --freeze_subsampling
    run_one_repeat "$dataset" "layer_drop_lr_sweep" "$lr" "drop-ctc-decoder" "$repeat" "$total_repeats" --freeze_ctc_decoder
    for layer in "${LAYERS[@]}"; do
        run_one_repeat "$dataset" "layer_drop_lr_sweep" "$lr" "drop-layer-${layer}" "$repeat" "$total_repeats" --freeze_layer "$layer"
    done
}

for repeat in "${EXTRA_REPEATS[@]}"; do
    for lr in "${EXTRA_ABLATION_EARNINGS22_LRS[@]}"; do
        for family in "${EXTRA_ABLATION_FAMILIES[@]}"; do
            run_extra_ablation_family "$family" "earnings22" "$lr" "$repeat" 3
        done
    done
    for lr in "${EXTRA_ABLATION_TEDLIUM_LRS[@]}"; do
        for family in "${EXTRA_ABLATION_FAMILIES[@]}"; do
            run_extra_ablation_family "$family" "tedlium" "$lr" "$repeat" 3
        done
    done
    for lr in "${LAYER_DROP_EARNINGS22_LRS[@]}"; do
        run_layer_drop_family "earnings22" "$lr" "$repeat" 3
    done
done

for repeat in "${LAYER_DROP_TEDLIUM_REPEATS[@]}"; do
    for lr in "${LAYER_DROP_TEDLIUM_LRS[@]}"; do
        run_layer_drop_family "tedlium" "$lr" "$repeat" 3
    done
done
