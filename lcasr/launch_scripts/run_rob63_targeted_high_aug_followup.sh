#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TRAINING_MODES="${TRAINING_MODES:-teacher_ce}"
export AUGS="${AUGS:-freq9_width44_time0}"
export FILTERS="${FILTERS:-no_filter}"
export RESULTS_ROOT="${RESULTS_ROOT:-./results/enc_dec/rob63_targeted_high_aug_followup}"

if [ "${RUN_REV16_TEST:-1}" = "1" ]; then
  DATASETS=rev16 \
  SPLIT=test \
  TRAINING_MODES="$TRAINING_MODES" \
  LRS="${REV16_LRS:-1e-7}" \
  AUGS="$AUGS" \
  FILTERS="$FILTERS" \
  RESULTS_ROOT="$RESULTS_ROOT" \
  bash launch_scripts/run_rob63_rl_self_training_compare.sh
fi

if [ "${RUN_EARNINGS22_TEST:-1}" = "1" ]; then
  DATASETS=earnings22 \
  SPLIT=test \
  TRAINING_MODES="$TRAINING_MODES" \
  LRS="${EARNINGS22_LRS:-3e-8}" \
  AUGS="$AUGS" \
  FILTERS="$FILTERS" \
  RESULTS_ROOT="$RESULTS_ROOT" \
  bash launch_scripts/run_rob63_rl_self_training_compare.sh
fi
