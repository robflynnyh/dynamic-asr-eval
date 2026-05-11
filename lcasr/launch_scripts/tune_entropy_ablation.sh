#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASETS_STR=${DATASETS:-"tedlium earnings22"}
SETTINGS_STR=${SETTINGS:-"freq_mask no_aug"}
SPLIT=${SPLIT:-test}
EPOCH=${EPOCH:-5}
SEQ=${SEQ:-16384}
OVERLAP=${OVERLAP:-14336}
REPEATS=${REPEATS:-1}
GPU=${GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
LR=${LR:-9e-5}
RESULTS_ROOT=${RESULTS_ROOT:-"./results/entropy_ablation"}
LOG_ROOT="${RESULTS_ROOT}/logs"
RAW_ROOT="${RESULTS_ROOT}/raw"
PKL_ROOT="${RESULTS_ROOT}/pkl"

read -r -a DATASETS <<< "$DATASETS_STR"
read -r -a SETTINGS <<< "$SETTINGS_STR"

mkdir -p "$LOG_ROOT" "$RAW_ROOT" "$PKL_ROOT"

run_item() {
    local dataset="$1"
    local setting="$2"

    local save_path="${PKL_ROOT}/${dataset}-${SPLIT}-epoch-${EPOCH}-${setting}.pkl"
    local trace_path="${RAW_ROOT}/${dataset}-${SPLIT}-epoch-${EPOCH}-${setting}.jsonl"
    local log_path="${LOG_ROOT}/${dataset}-${SPLIT}-epoch-${EPOCH}-${setting}.log"

    local checkpoint_args=()
    if [ -n "${CHECKPOINT:-}" ]; then
        checkpoint_args=(-c "$CHECKPOINT")
    fi

    local max_record_args=()
    if [ -n "${MAX_RECORDS:-}" ]; then
        max_record_args=(--max_records "$MAX_RECORDS")
    fi

    local aug_kwargs=()
    case "$setting" in
        freq_mask)
            aug_kwargs=(spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0)
            ;;
        no_aug)
            aug_kwargs=(spec_augment_n_freq_masks=0 spec_augment_freq_mask_param=0 spec_augment_n_time_masks=0)
            ;;
        *)
            echo "Unknown entropy ablation setting: $setting" >&2
            exit 1
            ;;
    esac

    {
        echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting entropy ablation dataset=${dataset} setting=${setting}"
        echo "gpu=${GPU}"
        echo "epochs=${EPOCH}"
        echo "save_path=${save_path}"
        echo "trace_path=${trace_path}"
        echo "log_path=${log_path}"
        echo "lr=${LR}"
        echo "kwargs=optim_lr=${LR} ${aug_kwargs[*]}"
    } | tee -a "$log_path"

    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_dynamic_eval_full.py \
        -dfa \
        -epochs "$EPOCH" \
        -seq "$SEQ" \
        -o "$OVERLAP" \
        -split "$SPLIT" \
        -d "$dataset" \
        -r "$REPEATS" \
        "${checkpoint_args[@]}" \
        "${max_record_args[@]}" \
        --entropy_trace_path "$trace_path" \
        --entropy_trace_setting "$setting" \
        -kwargs optim_lr="$LR" "${aug_kwargs[@]}" \
        -s "$save_path" 2>&1 | tee -a "$log_path"
}

for dataset in "${DATASETS[@]}"
do
    for setting in "${SETTINGS[@]}"
    do
        run_item "$dataset" "$setting"
    done
done

"$PYTHON_BIN" results/entropy_ablation/aggregate_entropy.py
"$PYTHON_BIN" results/entropy_ablation/plot_entropy.py --refresh
