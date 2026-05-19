#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_CHIME6_DEV=0
export RUN_REV16_TEST=1
export CHECKPOINTS="${CHECKPOINTS:-rl_step_30000}"
export TRAINING_MODES="${TRAINING_MODES:-teacher_ce}"
export LRS="${LRS:-3e-8}"
export AUGS="${AUGS:-freq9_width44_time0}"
export FILTERS="${FILTERS:-no_filter basic_repeat_filter}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export RESULTS_ROOT="${RESULTS_ROOT:-./results/enc_dec/checkpoint2/rob63_strong_aug_filter_followup}"

bash launch_scripts/run_rob63_strong_aug_filter_followup.sh
