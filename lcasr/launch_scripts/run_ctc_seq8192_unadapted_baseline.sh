#!/usr/bin/env bash
set -euo pipefail

# ROB-112: no-adapt CTC baseline for the 8192-context checkpoint.
# Run from lcasr/. Slurm and with-gpu launches should provide CUDA_VISIBLE_DEVICES.

DATASETS_STR=${DATASETS:-"earnings22 tedlium chime6 rev16"}
SPLIT=${SPLIT:-test}
SEQ=${SEQ:-8192}
OVERLAP=${OVERLAP:-7168}
REPEATS=${REPEATS:-1}
CHECKPOINT=${CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_8192_rp_1/step_105360.pt"}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
RESULTS_DIR=${RESULTS_DIR:-"./results/ctc_seq8192_unadapted_baseline"}
LOG_DIR="${RESULTS_DIR}/logs"
DRY_RUN=${DRY_RUN:-0}
MAX_RECORDS=${MAX_RECORDS:-}

read -r -a DATASETS <<< "$DATASETS_STR"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

run_eval() {
    local dataset="$1"
    local base_name="${dataset}-${SPLIT}-ctc-seq${SEQ}-overlap${OVERLAP}-no_adapt"
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
        -epochs 0
        -seq "$SEQ"
        -o "$OVERLAP"
        -split "$SPLIT"
        -d "$dataset"
        -r "$REPEATS"
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
    run_eval "$dataset"
done
