#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export DATASETS="${DATASETS:-chime6 rev16}"
export NEW_STEPS="${NEW_STEPS:-30000}"
export DECODE_CONFIGS="${DECODE_CONFIGS:-beam5_lp0p5}"
export RESULTS_ROOT="${RESULTS_ROOT:-./results/enc_dec/rl_step_30000/rob61_checkpoint_benchmark}"

bash launch_scripts/run_rob61_checkpoint_benchmark.sh
