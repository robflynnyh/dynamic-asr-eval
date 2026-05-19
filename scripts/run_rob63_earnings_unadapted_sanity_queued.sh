#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-63}"
SCREEN_NAME="${SCREEN_NAME:-rob63_earnings_unadapted_sanity}"
RESULTS_PATH="${RESULTS_PATH:-lcasr/results/enc_dec/checkpoint1/rob63_earnings_unadapted_sanity}"
LOG_PATH="${LOG_PATH:-${RESULTS_PATH}/screen.log}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob63_earnings_unadapted_sanity_queued.sh}"
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
    --note "ROB-63 Earnings22 unadapted sanity check exited. Inspect \`lcasr/results/enc_dec/checkpoint1/rob63_earnings_unadapted_sanity/OUTCOME.md\`, \`summary.csv\`, pickles, and logs before finalizing; this reruns beam5/lp0.5 unadapted Earnings22 for \`old_seed\` and the older \`enc_dec/OUTCOME\` checkpoint." \
    "${callback_args[@]}"
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${callback_status}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

mkdir -p "$(dirname "$LOG_PATH")" "$RESULTS_PATH"

if [ "${ROB63_EARNINGS_SANITY_SMOKE:-0}" = "1" ]; then
  {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-63 Earnings22 unadapted sanity wrapper smoke"
    echo "branch=${GIT_BRANCH}"
    echo "commit=${GIT_COMMIT}"
    echo "results_path=${RESULTS_PATH}"
  } 2>&1 | tee -a "$LOG_PATH"
  exit 0
fi

GPU="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"
export GPU

{
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-63 Earnings22 unadapted sanity check"
  echo "branch=${GIT_BRANCH}"
  echo "commit=${GIT_COMMIT}"
  echo "gpu=${GPU}"
  echo "old_checkpoint=${OLD_CHECKPOINT:-/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt}"
  echo "outcome_checkpoint=${OUTCOME_CHECKPOINT:-/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt}"
  echo "results_path=${RESULTS_PATH}"
} 2>&1 | tee -a "$LOG_PATH"

RESULTS_ROOT=./results/enc_dec/checkpoint1/rob63_earnings_unadapted_sanity \
bash lcasr/launch_scripts/run_rob63_earnings_unadapted_sanity.sh 2>&1 | tee -a "$LOG_PATH"
