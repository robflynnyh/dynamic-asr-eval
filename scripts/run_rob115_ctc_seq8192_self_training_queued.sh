#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-"/exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-115"}
if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE=${LINEAR_ISSUE:-ROB-115}
SCREEN_NAME=${SCREEN_NAME:-rob115_ctc_seq8192_self_training}
RESULTS_PATH=${RESULTS_PATH:-"${REPO_ROOT}/lcasr/results/ctc_seq8192_self_training_eval"}
LOG_PATH=${LOG_PATH:-"${RESULTS_PATH}/logs/${SCREEN_NAME}.log"}
RUNNER_LABEL=${RUNNER_LABEL:-"screen:${SCREEN_NAME}"}
QUEUED_COMMAND=${QUEUED_COMMAND:-"/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob115_ctc_seq8192_self_training_queued.sh"}
GIT_BRANCH=${GIT_BRANCH:-$(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}
GIT_COMMIT=${GIT_COMMIT:-$(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || printf 'unknown')}
CALLBACK_TARGET_STATE=${CALLBACK_TARGET_STATE:-Todo}
RUN_LR=${LR:-9e-5}
RUN_NOTE=${RUN_NOTE:-"ROB-115 8192-context CTC adapted freq-masking eval. Expected final outputs are 4 PKLs for earnings22, tedlium, chime6, and rev16 at seq_len=${SEQ:-8192} overlap=${OVERLAP:-7168} epochs=${EPOCHS:-5} lr=${RUN_LR}, plus regenerated summaries under lcasr/results/ctc_seq8192_self_training_eval."}

SCRATCH_ROOT=${SCRATCH_ROOT:-"${REPO_ROOT}/symphony/.scratch/ROB-115"}
mkdir -p "${SCRATCH_ROOT}/tmp" "${SCRATCH_ROOT}/mpl" "${SCRATCH_ROOT}/hf" "${SCRATCH_ROOT}/xdg"
export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_ROOT}/mpl}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/xdg}"

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
    --note "${RUN_NOTE}"
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
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-115 queued wrapper starting"
  echo "repo_root=${REPO_ROOT}"
  echo "branch=${GIT_BRANCH}"
  echo "commit=${GIT_COMMIT}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "results_path=${RESULTS_PATH}"
  echo "log_path=${LOG_PATH}"
  echo "callback_target_state=${CALLBACK_TARGET_STATE}"
  echo "tmpdir=${TMPDIR}"
  echo "checkpoint=${CHECKPOINT:-/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_8192_rp_1/step_105360.pt}"
  echo "seq=${SEQ:-8192}"
  echo "overlap=${OVERLAP:-7168}"
  echo "epochs=${EPOCHS:-5}"
  echo "lr=${RUN_LR}"
  echo "datasets=${DATASETS:-earnings22 tedlium chime6 rev16}"
} | tee -a "$LOG_PATH"

if [ "${CALLBACK_SMOKE_ONLY:-0}" = "1" ]; then
  echo "CALLBACK_SMOKE_ONLY=1: exiting through wrapper EXIT trap without running GPU eval." | tee -a "$LOG_PATH"
  exit 0
fi

cd "$REPO_ROOT/lcasr"
PYTHON_BIN=${PYTHON_BIN:-python3.10} \
RESULTS_DIR="./results/ctc_seq8192_self_training_eval" \
CHECKPOINT=${CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_8192_rp_1/step_105360.pt"} \
DATASETS=${DATASETS:-"earnings22 tedlium chime6 rev16"} \
EPOCHS=${EPOCHS:-"5"} \
REPEATS=${REPEATS:-1} \
SEQ=${SEQ:-8192} \
OVERLAP=${OVERLAP:-7168} \
LR=${RUN_LR} \
bash launch_scripts/run_ctc_seq8192_self_training_eval.sh 2>&1 | tee -a "$LOG_PATH"

cd "$REPO_ROOT"
python lcasr/results/ctc_seq8192_self_training_eval/aggregate.py 2>&1 | tee -a "$LOG_PATH"
