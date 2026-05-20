#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-97}"
NOTIFY_ISSUE="${NOTIFY_ISSUE:-ROB-96}"
SCREEN_NAME="${SCREEN_NAME:-rob97_encdec_v2_unadapted_beam}"
RESULTS_PATH="${RESULTS_PATH:-lcasr/results/enc_dec/enc_dec_v2/rob97_unadapted_beam}"
LOG_PATH="${LOG_PATH:-${RESULTS_PATH}/screen.log}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob97_enc_dec_v2_unadapted_beam_queued.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

post_callback() {
  local issue="$1"
  local status="$2"
  local target_state="$3"
  local note="$4"
  callback_args=()
  if [ "${CALLBACK_DRY_RUN:-0}" = "1" ]; then
    callback_args+=(--dry-run)
  fi
  python3 scripts/linear_experiment_callback.py \
    --issue "${issue}" \
    --status-code "${status}" \
    --log "${LOG_PATH}" \
    --results "${RESULTS_PATH}" \
    --screen-name "${SCREEN_NAME}" \
    --runner-label "${RUNNER_LABEL}" \
    --queued-command "${QUEUED_COMMAND}" \
    --branch "${GIT_BRANCH}" \
    --commit "${GIT_COMMIT}" \
    --target-state "${target_state}" \
    --max-log-chars 20000 \
    --max-body-chars 60000 \
    --note "${note}" \
    "${callback_args[@]}"
}

on_exit() {
  status=$?
  set +e
  if [ -z "${LINEAR_API_KEY:-}" ] && [ "${CALLBACK_DRY_RUN:-0}" != "1" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "${status}"
  fi

  post_callback \
    "${LINEAR_ISSUE}" \
    "${status}" \
    "Todo" \
    "ROB-97 enc_dec_v2 no-adapt beam-search wrapper completed. Inspect \`lcasr/results/enc_dec/enc_dec_v2/rob97_unadapted_beam/OUTCOME.md\`, \`summary.csv\`, PKLs, and logs before finalizing. The target rows are \`chime6/test\` and \`rev16/test\` with beam_width=5 and length_penalty=0.5."
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "ROB-97 Linear completion callback failed with status ${callback_status}" >&2
  fi

  if [ "${status}" -eq 0 ]; then
    post_callback \
      "${NOTIFY_ISSUE}" \
      "${status}" \
      "Todo" \
      "ROB-97 completed the missing enc_dec_v2 before-adaptation beam-search evals for ROB-96. Incorporate the \`chime6/test\` and \`rev16/test\` rows from \`lcasr/results/enc_dec/enc_dec_v2/rob97_unadapted_beam/summary.csv\`."
    notify_status=$?
    if [ "${notify_status}" -ne 0 ]; then
      echo "${NOTIFY_ISSUE} notification callback failed with status ${notify_status}" >&2
    fi
  fi
  exit "${status}"
}
trap on_exit EXIT

mkdir -p "$(dirname "$LOG_PATH")" "$RESULTS_PATH"

if [ "${ROB97_SMOKE:-0}" = "1" ]; then
  {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-97 enc_dec_v2 unadapted beam wrapper smoke"
    echo "branch=${GIT_BRANCH}"
    echo "commit=${GIT_COMMIT}"
    echo "results_path=${RESULTS_PATH}"
  } 2>&1 | tee -a "$LOG_PATH"
  exit 0
fi

GPU="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"
export GPU

{
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-97 enc_dec_v2 unadapted beam-search evals"
  echo "branch=${GIT_BRANCH}"
  echo "commit=${GIT_COMMIT}"
  echo "gpu=${GPU}"
  echo "checkpoint=/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt"
  echo "rows=chime6/test rev16/test"
  echo "decode=beam5_lp0p5"
  echo "epochs=0"
  echo "results_path=${RESULTS_PATH}"
} 2>&1 | tee -a "$LOG_PATH"

RESULTS_DIR=./results/enc_dec/enc_dec_v2/rob97_unadapted_beam \
SKIP_EXISTING="${SKIP_EXISTING:-1}" \
bash lcasr/launch_scripts/run_rob97_enc_dec_v2_unadapted_beam.sh 2>&1 | tee -a "$LOG_PATH"
