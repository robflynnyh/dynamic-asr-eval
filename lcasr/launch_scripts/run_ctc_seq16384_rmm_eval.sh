#!/usr/bin/env bash
set -euo pipefail

# ROB-68: CTC self-training eval using the RMM mixed-mask augmentation policy
# with the normal 16384-context setup. Run from lcasr/. When launched through
# with-gpu, inherit its CUDA_VISIBLE_DEVICES assignment; set GPU manually only
# for local smoke runs.

DATASETS_STR=${DATASETS:-"earnings22 tedlium chime6 rev16"}
EPOCHS_STR=${EPOCHS:-"1 5"}
SPLIT=${SPLIT:-test}
SEQ=${SEQ:-16384}
OVERLAP=${OVERLAP:-14336}
REPEATS=${REPEATS:-1}
TOTAL_REPEATS=${TOTAL_REPEATS:-$REPEATS}
REPEAT_INDICES_STR=${REPEAT_INDICES:-}
SKIP_EXISTING=${SKIP_EXISTING:-1}
LR=${LR:-9e-5}
LRS_STR=${LRS:-"$LR"}
CHECKPOINT=${CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_16384_rp_1/step_105360.pt"}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
RESULTS_DIR=${RESULTS_DIR:-"./results/ctc_seq16384_rmm_eval"}
LOG_DIR="${RESULTS_DIR}/logs"
DRY_RUN=${DRY_RUN:-0}
MAX_RECORDS=${MAX_RECORDS:-}

read -r -a DATASETS <<< "$DATASETS_STR"
read -r -a EPOCHS <<< "$EPOCHS_STR"
read -r -a LRS_ARR <<< "$LRS_STR"
read -r -a REPEAT_INDICES_ARR <<< "$REPEAT_INDICES_STR"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

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
if isinstance(data, dict):
    data["repeat"] = f"{repeat}/{total_repeats}"
    if isinstance(data.get("args_dict"), dict):
        data["args_dict"]["repeats"] = int(total_repeats)
        data["args_dict"]["save_path"] = base_save_path
with path.open("wb") as f:
    pickle.dump(data, f)
PY
}

run_single_repeat() {
    local dataset="$1"
    local epoch="$2"
    local lr="$3"
    local repeat="$4"
    local total_repeats="$5"
    local lr_slug
    lr_slug=$(lr_tag "$lr")

    local base_name="${dataset}-${SPLIT}-ctc-seq${SEQ}-overlap${OVERLAP}-rmm-epoch-${epoch}-lr-${lr_slug}"
    local base_save_path="${RESULTS_DIR}/${base_name}.pkl"
    local target_path="${RESULTS_DIR}/${base_name}_${repeat}.pkl"
    local tmp_base="${RESULTS_DIR}/${base_name}.repeat-${repeat}.tmp.pkl"
    local tmp_output="${tmp_base%.pkl}_1.pkl"
    local log_path="${LOG_DIR}/${base_name}_repeat-${repeat}.log"

    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$target_path" ]; then
        echo "Skipping existing ${target_path}" | tee -a "$log_path"
        return
    fi

    rm -f "$tmp_output"
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${base_name} repeat=${repeat}/${total_repeats}" | tee -a "$log_path"
    echo "checkpoint=${CHECKPOINT}" | tee -a "$log_path"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$log_path"
    echo "manual_gpu=${GPU:-unset}" | tee -a "$log_path"
    echo "target_path=${target_path}" | tee -a "$log_path"
    echo "augmentation_policy=rmm" | tee -a "$log_path"

    local cmd=(
        "$PYTHON_BIN" run_dynamic_eval_full.py
        -c "$CHECKPOINT"
        -dfa
        -epochs "$epoch"
        -seq "$SEQ"
        -o "$OVERLAP"
        -split "$SPLIT"
        -d "$dataset"
        -r 1
        -kwargs
        optim_lr="$lr"
        augmentation_policy="'rmm'"
        -s "$tmp_base"
    )

    if [ -n "$MAX_RECORDS" ]; then
        cmd+=(--max_records "$MAX_RECORDS")
    fi

    if [ "$DRY_RUN" = "1" ]; then
        printf 'DRY_RUN'
        if [ -n "${GPU:-}" ]; then
            printf ' CUDA_VISIBLE_DEVICES=%q' "$GPU"
        fi
        printf ' %q' "${cmd[@]}"
        printf ' && patch_repeat_metadata %q %q %q %q && mv %q %q' \
            "$tmp_output" "$base_save_path" "$repeat" "$total_repeats" "$tmp_output" "$target_path"
        printf '\n' | tee -a "$log_path"
        return
    fi

    if [ -n "${GPU:-}" ]; then
        CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" 2>&1 | tee -a "$log_path"
    else
        "${cmd[@]}" 2>&1 | tee -a "$log_path"
    fi

    patch_repeat_metadata "$tmp_output" "$base_save_path" "$repeat" "$total_repeats"
    mv "$tmp_output" "$target_path"
    echo "Saved repeat ${repeat}/${total_repeats} to ${target_path}" | tee -a "$log_path"
}

run_eval() {
    local dataset="$1"
    local epoch="$2"
    local lr="$3"
    local lr_slug
    lr_slug=$(lr_tag "$lr")

    local base_name="${dataset}-${SPLIT}-ctc-seq${SEQ}-overlap${OVERLAP}-rmm-epoch-${epoch}-lr-${lr_slug}"
    local save_path="${RESULTS_DIR}/${base_name}.pkl"
    local log_path="${LOG_DIR}/${base_name}.log"

    if [ -n "$REPEAT_INDICES_STR" ]; then
        for repeat in "${REPEAT_INDICES_ARR[@]}"; do
            run_single_repeat "$dataset" "$epoch" "$lr" "$repeat" "$TOTAL_REPEATS"
        done
        return
    fi

    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${base_name}" | tee -a "$log_path"
    echo "checkpoint=${CHECKPOINT}" | tee -a "$log_path"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$log_path"
    echo "manual_gpu=${GPU:-unset}" | tee -a "$log_path"
    echo "save_path=${save_path}" | tee -a "$log_path"
    echo "repeats=${REPEATS}" | tee -a "$log_path"
    echo "augmentation_policy=rmm" | tee -a "$log_path"

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
        optim_lr="$lr"
        augmentation_policy="'rmm'"
        -s "$save_path"
    )

    if [ -n "$MAX_RECORDS" ]; then
        cmd+=(--max_records "$MAX_RECORDS")
    fi

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
        for lr in "${LRS_ARR[@]}"; do
            run_eval "$dataset" "$epoch" "$lr"
        done
    done
done
