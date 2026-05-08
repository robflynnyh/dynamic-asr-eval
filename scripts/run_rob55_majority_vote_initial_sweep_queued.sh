#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-55}"
SCREEN_NAME="${SCREEN_NAME:-rob55_majority_vote_initial}"
RESULTS_PATH="${RESULTS_PATH:-lcasr/results/enc_dec/enc_dec_majority_vote}"
LOG_PATH="${LOG_PATH:-${RESULTS_PATH}/logs/rob55_majority_vote_initial.log}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob55_majority_vote_initial_sweep_queued.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
  if [ -z "${LINEAR_API_KEY:-}" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "${status}"
  fi
  python3 scripts/linear_experiment_callback.py \
    --issue "${LINEAR_ISSUE}" \
    --status-code "${status}" \
    --log "${LOG_PATH}" \
    --results "${RESULTS_PATH}" \
    --screen-name "${SCREEN_NAME}" \
    --runner-label "screen:${SCREEN_NAME}" \
    --queued-command "${QUEUED_COMMAND}" \
    --branch "${GIT_BRANCH}" \
    --commit "${GIT_COMMIT}" \
    --target-state Todo \
    --note "ROB-55 initial bounded majority-vote encoder-decoder self-training sweep finished. Inspect the result pickles and aggregate summary before deciding whether to expand the grid."
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${callback_status}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

set -euo pipefail

mkdir -p "$(dirname "$LOG_PATH")" "$RESULTS_PATH"
{
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-55 majority-vote initial sweep"
  echo "repo=${REPO_ROOT}"
  echo "branch=${GIT_BRANCH}"
  echo "commit=${GIT_COMMIT}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "queued_command=${QUEUED_COMMAND}"
} | tee "$LOG_PATH"

cd lcasr
GPU="${CUDA_VISIBLE_DEVICES:-0}" \
RESULTS_DIR="./results/enc_dec/enc_dec_majority_vote" \
RUN_BASELINE=1 \
bash launch_scripts/tune_enc_dec_majority_vote_tedlium_dev.sh 2>&1 | tee -a "../${LOG_PATH}"

python results/enc_dec/enc_dec_majority_vote/aggregate.py \
  --csv results/enc_dec/enc_dec_majority_vote/summary.csv \
  2>&1 | tee -a "../${LOG_PATH}"
