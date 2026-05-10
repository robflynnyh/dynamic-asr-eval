#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-"/exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-56"}
if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE=${LINEAR_ISSUE:-ROB-56}
SCREEN_NAME=${SCREEN_NAME:-rob56_ctc_seq2048_selected_lr_all_datasets}
RESULTS_PATH=${RESULTS_PATH:-"${REPO_ROOT}/lcasr/results/ctc_seq2048_self_training_final_lr"}
LOG_PATH=${LOG_PATH:-"${RESULTS_PATH}/logs/${SCREEN_NAME}.log"}
RUNNER_LABEL=${RUNNER_LABEL:-"screen:${SCREEN_NAME}"}
QUEUED_COMMAND=${QUEUED_COMMAND:-"/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob56_ctc_seq2048_selected_lr_queued.sh"}
GIT_BRANCH=${GIT_BRANCH:-$(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}
GIT_COMMIT=${GIT_COMMIT:-$(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || printf 'unknown')}
CALLBACK_TARGET_STATE=${CALLBACK_TARGET_STATE:-Todo}

on_exit() {
  status=$?
  set +e
  mkdir -p "$(dirname "${LOG_PATH}")"
  cd "$REPO_ROOT" || exit "$status"

  callback_args=(
    --issue "${LINEAR_ISSUE}"
    --status-code "${status}"
    --log "${LOG_PATH}"
    --results "${RESULTS_PATH}"
    --screen-name "${SCREEN_NAME}"
    --runner-label "${RUNNER_LABEL}"
    --queued-command "${QUEUED_COMMAND}"
    --branch "${GIT_BRANCH}"
    --commit "${GIT_COMMIT}"
    --target-state "${CALLBACK_TARGET_STATE}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-20000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-60000}"
    --note "ROB-56 final all-dataset selected-LR run for the CTC 2048-context self-training eval. Epoch 1 uses lr=1e-5; epoch 5 uses lr=1e-6, selected from the TEDLIUM lower-LR pilot."
  )

  if [ "${CALLBACK_DRY_RUN:-0}" = "1" ]; then
    callback_args+=(--dry-run)
  fi

  if [ -z "${LINEAR_API_KEY:-}" ] && [ "${CALLBACK_DRY_RUN:-0}" != "1" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "$status"
  fi

  python3 "${REPO_ROOT}/scripts/linear_experiment_callback.py" "${callback_args[@]}"
  callback_status=$?
  if [ "$callback_status" -ne 0 ]; then
    echo "Linear completion callback failed with status ${callback_status}" >&2
  fi
  exit "$status"
}
trap on_exit EXIT

set -euo pipefail

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULTS_PATH}"
cd "$REPO_ROOT"

{
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-56 selected-LR all-dataset wrapper starting"
  echo "repo_root=${REPO_ROOT}"
  echo "branch=${GIT_BRANCH}"
  echo "commit=${GIT_COMMIT}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "results_path=${RESULTS_PATH}"
  echo "log_path=${LOG_PATH}"
  echo "callback_target_state=${CALLBACK_TARGET_STATE}"
  echo "epoch1_lr=${EPOCH1_LR:-1e-5}"
  echo "epoch5_lr=${EPOCH5_LR:-1e-6}"
} | tee -a "$LOG_PATH"

if [ "${CALLBACK_SMOKE_ONLY:-0}" = "1" ]; then
  echo "CALLBACK_SMOKE_ONLY=1: exiting through wrapper EXIT trap without running GPU eval." | tee -a "$LOG_PATH"
  exit 0
fi

cd "$REPO_ROOT/lcasr"

COMMON_ENV=(
  PYTHON_BIN=${PYTHON_BIN:-python3.10}
  RESULTS_DIR="./results/ctc_seq2048_self_training_final_lr"
  CHECKPOINT=${CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt"}
  DATASETS=${DATASETS:-"earnings22 tedlium chime6 rev16"}
  REPEATS=${REPEATS:-1}
  SEQ=${SEQ:-2048}
  OVERLAP=${OVERLAP:-1792}
)

env "${COMMON_ENV[@]}" EPOCHS=1 LR=${EPOCH1_LR:-1e-5} \
  bash launch_scripts/run_ctc_seq2048_self_training_eval.sh 2>&1 | tee -a "$LOG_PATH"

env "${COMMON_ENV[@]}" EPOCHS=5 LR=${EPOCH5_LR:-1e-6} \
  bash launch_scripts/run_ctc_seq2048_self_training_eval.sh 2>&1 | tee -a "$LOG_PATH"
