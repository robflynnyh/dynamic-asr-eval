#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-63}"
SCREEN_NAME="${SCREEN_NAME:-rob63_remaining_normal_baselines}"
RESULTS_PATH="${RESULTS_PATH:-lcasr/results/enc_dec/checkpoint2/rob61_checkpoint_benchmark}"
LOG_PATH="${LOG_PATH:-${RESULTS_PATH}/rob63_remaining_normal_baselines_screen.log}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob63_remaining_normal_baselines_queued.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
  callback_args=()
  if [ "${CALLBACK_DRY_RUN:-0}" = "1" ]; then
    callback_args+=(--dry-run)
  fi
  if [ -z "${LINEAR_API_KEY:-}" ] && [ "${CALLBACK_DRY_RUN:-0}" != "1" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "${status}"
  fi
  python3 scripts/linear_experiment_callback.py \
    --issue "${LINEAR_ISSUE}" \
    --status-code "${status}" \
    --log "${LOG_PATH}" \
    --results "${RESULTS_PATH}" \
    --screen-name "${SCREEN_NAME}" \
    --runner-label "${RUNNER_LABEL}" \
    --queued-command "${QUEUED_COMMAND}" \
    --branch "${GIT_BRANCH}" \
    --commit "${GIT_COMMIT}" \
    --target-state Todo \
    --note "ROB-63 remaining normal-baseline wrapper completed. Inspect \`lcasr/results/enc_dec/checkpoint2/rob61_checkpoint_benchmark/OUTCOME.md\`, \`summary.csv\`, new CHiME-6/Rev16 normal pickles, and logs before regenerating ROB-63 summaries." \
    "${callback_args[@]}"
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${callback_status}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

mkdir -p "$(dirname "$LOG_PATH")" "$RESULTS_PATH"

if [ "${ROB63_BASELINE_SMOKE:-0}" = "1" ]; then
  {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-63 remaining normal-baseline wrapper smoke"
    echo "branch=${GIT_BRANCH}"
    echo "commit=${GIT_COMMIT}"
    echo "results_path=${RESULTS_PATH}"
  } 2>&1 | tee -a "$LOG_PATH"
  exit 0
fi

GPU="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"
export GPU

{
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-63 remaining normal baselines"
  echo "branch=${GIT_BRANCH}"
  echo "commit=${GIT_COMMIT}"
  echo "gpu=${GPU}"
  echo "datasets=${DATASETS:-chime6 rev16}"
  echo "new_steps=${NEW_STEPS:-30000}"
  echo "decode_configs=${DECODE_CONFIGS:-beam5_lp0p5}"
  echo "results_path=${RESULTS_PATH}"
} 2>&1 | tee -a "$LOG_PATH"

DATASETS="${DATASETS:-chime6 rev16}" \
NEW_STEPS="${NEW_STEPS:-30000}" \
DECODE_CONFIGS="${DECODE_CONFIGS:-beam5_lp0p5}" \
RESULTS_ROOT=./results/enc_dec/checkpoint2/rob61_checkpoint_benchmark \
bash lcasr/launch_scripts/run_rob63_remaining_normal_baselines.sh 2>&1 | tee -a "$LOG_PATH"
