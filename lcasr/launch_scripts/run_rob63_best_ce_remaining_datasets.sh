#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export DATASETS="${DATASETS:-chime6 rev16}"
export TRAINING_MODES="${TRAINING_MODES:-teacher_ce}"
export LRS="${LRS:-1e-7}"
export AUGS="${AUGS:-freq3_width24_time0}"
export RESULTS_ROOT="${RESULTS_ROOT:-./results/enc_dec/rob63_best_ce_remaining_datasets}"

bash launch_scripts/run_rob63_rl_self_training_compare.sh
