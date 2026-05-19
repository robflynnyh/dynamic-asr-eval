#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TRAINING_MODES="${TRAINING_MODES:-teacher_ce}"
export LRS="${LRS:-1e-8 3e-8}"
export AUGS="${AUGS:-freq9_width44_time0}"
export FILTERS="${FILTERS:-no_filter basic_repeat_filter}"
export RESULTS_ROOT="${RESULTS_ROOT:-./results/enc_dec/checkpoint2/rob63_strong_aug_filter_followup}"

if [ "${RUN_CHIME6_DEV:-1}" = "1" ]; then
  DATASETS=chime6 \
  SPLIT=dev \
  TRAINING_MODES="$TRAINING_MODES" \
  LRS="$LRS" \
  AUGS="$AUGS" \
  FILTERS="$FILTERS" \
  RESULTS_ROOT="$RESULTS_ROOT" \
  bash launch_scripts/run_rob63_rl_self_training_compare.sh
fi

if [ "${RUN_REV16_TEST:-1}" = "1" ]; then
  DATASETS=rev16 \
  SPLIT=test \
  TRAINING_MODES="$TRAINING_MODES" \
  LRS="$LRS" \
  AUGS="$AUGS" \
  FILTERS="$FILTERS" \
  RESULTS_ROOT="$RESULTS_ROOT" \
  bash launch_scripts/run_rob63_rl_self_training_compare.sh
fi
