#!/usr/bin/env bash
# Template for long-running dynamic-asr-eval GPU experiments launched through
# with-gpu. Copy this file, fill in the variables below, and keep the EXIT trap
# intact.

set -uo pipefail

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-000}"
LOG_PATH="${LOG_PATH:-lcasr/results/example/logs/run.log}"
RESULTS_PATH="${RESULTS_PATH:-lcasr/results/example}"
SCREEN_NAME="${SCREEN_NAME:-dynamic_asr_eval_example_run}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/queued_experiment_wrapper.sh}"
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
    --runner-label "${RUNNER_LABEL}" \
    --queued-command "${QUEUED_COMMAND}" \
    --branch "${GIT_BRANCH}" \
    --commit "${GIT_COMMIT}"
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${callback_status}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

set -euo pipefail

# Replace this block with the real experiment command. Keep output tee'd or
# redirected to LOG_PATH so the callback can include useful failure evidence.
mkdir -p "$(dirname "${LOG_PATH}")" "${RESULTS_PATH}"
python3 - <<'PY' 2>&1 | tee -a "${LOG_PATH}"
from pathlib import Path

Path("lcasr/results/example/hello.txt").write_text("hello world\n")
print("dynamic-asr-eval queued experiment wrapper smoke completed")
PY
