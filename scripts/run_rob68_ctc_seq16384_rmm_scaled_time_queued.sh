#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-"/exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-68"}
if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE=${LINEAR_ISSUE:-ROB-68}
SCREEN_NAME=${SCREEN_NAME:-rob68_ctc_seq16384_rmm_scaled_time_masks}
RESULTS_PATH=${RESULTS_PATH:-"${REPO_ROOT}/lcasr/results/ctc_seq16384_rmm_scaled_time_masks_eval"}
LOG_PATH=${LOG_PATH:-"${RESULTS_PATH}/logs/${SCREEN_NAME}.log"}
RUNNER_LABEL=${RUNNER_LABEL:-"screen:${SCREEN_NAME}"}
QUEUED_COMMAND=${QUEUED_COMMAND:-"/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob68_ctc_seq16384_rmm_scaled_time_queued.sh"}
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
    --note "ROB-68 scaled-time-mask RMM follow-up (${SPLIT:-test} split; datasets: ${DATASETS:-earnings22 tedlium chime6 rev16}): 16384-context RMM with 2048-like mask widths and scaled time-mask count. Inspect lcasr/results/ctc_seq16384_rmm_scaled_time_masks_eval summary files and logs before finalizing."
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
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-68 scaled-time-mask RMM eval wrapper starting"
  echo "repo_root=${REPO_ROOT}"
  echo "branch=${GIT_BRANCH}"
  echo "commit=${GIT_COMMIT}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "results_path=${RESULTS_PATH}"
  echo "log_path=${LOG_PATH}"
  echo "callback_target_state=${CALLBACK_TARGET_STATE}"
  echo "split=${SPLIT:-test}"
  echo "datasets=${DATASETS:-earnings22 tedlium chime6 rev16}"
  echo "lr=${LR:-9e-5}"
  echo "repeats=${REPEATS:-3}"
  echo "rmm_scale_time_masks_by_seq_len=True"
  echo "rmm_time_masks_reference_seq_len=${RMM_TIME_MASKS_REFERENCE_SEQ_LEN:-2048}"
} | tee -a "$LOG_PATH"

if [ "${CALLBACK_SMOKE_ONLY:-0}" = "1" ]; then
  echo "CALLBACK_SMOKE_ONLY=1: exiting through wrapper EXIT trap without running GPU eval." | tee -a "$LOG_PATH"
  exit 0
fi

cd "$REPO_ROOT/lcasr"

COMMON_ENV=(
  PYTHON_BIN=${PYTHON_BIN:-python3.10}
  RESULTS_DIR="./results/ctc_seq16384_rmm_scaled_time_masks_eval"
  CHECKPOINT=${CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_16384_rp_1/step_105360.pt"}
  DATASETS=${DATASETS:-"earnings22 tedlium chime6 rev16"}
  REPEATS=${REPEATS:-3}
  SEQ=${SEQ:-16384}
  OVERLAP=${OVERLAP:-14336}
  AUGMENTATION_LABEL=${AUGMENTATION_LABEL:-rmm-width2048-scaled}
  RMM_SCALE_TIME_MASKS_BY_SEQ_LEN=True
  RMM_TIME_MASKS_REFERENCE_SEQ_LEN=${RMM_TIME_MASKS_REFERENCE_SEQ_LEN:-2048}
)

env "${COMMON_ENV[@]}" EPOCHS="${EPOCHS:-1 5}" LR=${LR:-9e-5} \
  bash launch_scripts/run_ctc_seq16384_rmm_eval.sh 2>&1 | tee -a "$LOG_PATH"

cd "$REPO_ROOT"
python lcasr/results/ctc_seq16384_rmm_scaled_time_masks_eval/aggregate.py 2>&1 | tee -a "$LOG_PATH"
