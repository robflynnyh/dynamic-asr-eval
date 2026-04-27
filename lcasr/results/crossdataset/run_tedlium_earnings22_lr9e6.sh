#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3.10}
DEVICE=${DEVICE:-3}
EPOCHS=(1 5)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

for epoch in "${EPOCHS[@]}"
do
    save_path="./results/crossdataset/tedlium_earnings22-epoch-${epoch}-lr9e6-test.pkl"
    log_path="./results/crossdataset/logs/tedlium_earnings22-epoch-${epoch}-lr9e6-test.log"

    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting tedlium -> earnings22 cross-dataset eval"
    echo "epoch=${epoch}"
    echo "save_path=${save_path}"
    echo "log_path=${log_path}"

    CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON_BIN" run_cross_dataset_eval.py \
        -dfa \
        -epochs "$epoch" \
        -seq 16384 \
        -o 14336 \
        -split test \
        -d tedlium \
        -d2 earnings22 \
        -r 3 \
        -kwargs optim_lr=0.000009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 \
        -s "$save_path" 2>&1 | tee "$log_path"

    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] finished epoch ${epoch}"
done
