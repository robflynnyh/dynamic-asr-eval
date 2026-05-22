#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-"/exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-110"}
if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE=${LINEAR_ISSUE:-ROB-110}
SCREEN_NAME=${SCREEN_NAME:-rob110_ctc_seq16384_unadapted_repeats23}
RESULTS_PATH=${RESULTS_PATH:-"${REPO_ROOT}/lcasr/results/ctc_seq16384_unadapted_baseline_repeats"}
LOG_PATH=${LOG_PATH:-"${RESULTS_PATH}/logs/${SCREEN_NAME}.log"}
RUNNER_LABEL=${RUNNER_LABEL:-"screen:${SCREEN_NAME}"}
QUEUED_COMMAND=${QUEUED_COMMAND:-"/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob110_ctc_seq16384_unadapted_repeats_queued.sh"}
GIT_BRANCH=${GIT_BRANCH:-$(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}
GIT_COMMIT=${GIT_COMMIT:-$(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || printf 'unknown')}
CALLBACK_TARGET_STATE=${CALLBACK_TARGET_STATE:-Todo}
SCRATCH_ROOT=${SCRATCH_ROOT:-"${REPO_ROOT}/symphony/.scratch/rob110_repeats"}
CHECKPOINT_SPECS=${CHECKPOINT_SPECS:-"rp_2=/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_16384_rp_2/step_105360.pt rp_3=/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_16384_rp_3/step_105360.pt"}

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
    --note "ROB-110 follow-up repeat check: run no-adapt epoch-0 16384-context CTC baselines for n_seq_sched_16384_rp_2 and rp_3 without overwriting the committed rp_1 artifacts. Inspect lcasr/results/ctc_seq16384_unadapted_baseline_repeats/ and rerun the repeat comparison before finalizing."
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

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULTS_PATH}" "${SCRATCH_ROOT}"/{tmp,cache,matplotlib,hf}
export TMPDIR="${SCRATCH_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib"
export HF_HOME="${SCRATCH_ROOT}/hf"

cd "$REPO_ROOT"

{
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-110 unadapted repeat wrapper starting"
  echo "repo_root=${REPO_ROOT}"
  echo "branch=${GIT_BRANCH}"
  echo "commit=${GIT_COMMIT}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "results_path=${RESULTS_PATH}"
  echo "log_path=${LOG_PATH}"
  echo "scratch_root=${SCRATCH_ROOT}"
  echo "callback_target_state=${CALLBACK_TARGET_STATE}"
  echo "checkpoint_specs=${CHECKPOINT_SPECS}"
} | tee -a "$LOG_PATH"

if [ "${CALLBACK_SMOKE_ONLY:-0}" = "1" ]; then
  echo "CALLBACK_SMOKE_ONLY=1: exiting through wrapper EXIT trap without running GPU eval." | tee -a "$LOG_PATH"
  exit 0
fi

cd "$REPO_ROOT/lcasr"

for spec in ${CHECKPOINT_SPECS}; do
  label=${spec%%=*}
  checkpoint=${spec#*=}
  result_dir="./results/ctc_seq16384_unadapted_baseline_repeats/${label}"

  if [ ! -f "$checkpoint" ]; then
    echo "Missing checkpoint for ${label}: ${checkpoint}" | tee -a "$LOG_PATH"
    exit 1
  fi

  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${label}: ${checkpoint}" | tee -a "$LOG_PATH"
  env \
    PYTHON_BIN=${PYTHON_BIN:-python3.10} \
    RESULTS_DIR="${result_dir}" \
    CHECKPOINT="${checkpoint}" \
    DATASETS="${DATASETS:-earnings22 tedlium chime6 rev16}" \
    REPEATS="${REPEATS:-1}" \
    SEQ="${SEQ:-16384}" \
    OVERLAP="${OVERLAP:-14336}" \
    ${MAX_RECORDS:+MAX_RECORDS="${MAX_RECORDS}"} \
    bash launch_scripts/run_ctc_seq16384_unadapted_baseline.sh 2>&1 | tee -a "$LOG_PATH"

  cd "$REPO_ROOT"
  python lcasr/results/ctc_seq16384_unadapted_baseline/aggregate.py \
    --root "lcasr/results/ctc_seq16384_unadapted_baseline_repeats/${label}" 2>&1 | tee -a "$LOG_PATH"
  cd "$REPO_ROOT/lcasr"
done

cd "$REPO_ROOT"
python lcasr/results/ctc_seq16384_unadapted_baseline/compare_checkpoint_repeats.py 2>&1 | tee -a "$LOG_PATH"
