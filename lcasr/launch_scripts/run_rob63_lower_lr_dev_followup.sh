#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TRAINING_MODES="${TRAINING_MODES:-teacher_ce}"
export LRS="${LRS:-3e-8 1e-8 3e-9}"
export AUGS="${AUGS:-freq3_width24_time0}"
export RESULTS_ROOT="${RESULTS_ROOT:-./results/enc_dec/rob63_lower_lr_dev_followup}"

if [ "${RUN_CHIME6_DEV:-1}" = "1" ]; then
  DATASETS=chime6 \
  SPLIT=dev \
  TRAINING_MODES="$TRAINING_MODES" \
  LRS="$LRS" \
  AUGS="$AUGS" \
  RESULTS_ROOT="$RESULTS_ROOT" \
  bash launch_scripts/run_rob63_rl_self_training_compare.sh
fi

if [ "${RUN_REV16_TEST:-1}" = "1" ]; then
  DATASETS=rev16 \
  SPLIT=test \
  TRAINING_MODES="$TRAINING_MODES" \
  LRS="$LRS" \
  AUGS="$AUGS" \
  RESULTS_ROOT="$RESULTS_ROOT" \
  bash launch_scripts/run_rob63_rl_self_training_compare.sh
fi
