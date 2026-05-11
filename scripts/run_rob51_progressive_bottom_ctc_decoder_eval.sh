#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-c16a3e06-e950-47c1-b561-017411138f47}"
SCREEN_NAME="${SCREEN_NAME:-rob51_progressive_bottom_ctc_decoder}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
RESULTS_ROOT="${RESULTS_ROOT:-./results/ctc_self_training_extra_ablation_sweeps}"
RESULTS_PATH="${RESULTS_PATH:-${REPO_ROOT}/lcasr/results/ctc_self_training_extra_ablation_sweeps/progressive_bottom_ctc_decoder}"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/lcasr/results/ctc_self_training_extra_ablation_sweeps/logs/rob51_progressive_bottom_ctc_decoder_eval.log}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob51_progressive_bottom_ctc_decoder_eval.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
CALLBACK_SCRIPT="${CALLBACK_SCRIPT:-${REPO_ROOT}/scripts/linear_experiment_callback.py}"

on_exit() {
  status=$?
  trap - EXIT
  set +e
  if [ -z "${LINEAR_API_KEY:-}" ] && [ "${CALLBACK_DRY_RUN:-0}" != "1" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "${status}"
  fi

  callback_args=(
    "${CALLBACK_SCRIPT}"
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
    --note "ROB-51 progressive_bottom_ctc_decoder run. On success, inspect the new PKLs, summary CSV/Markdown, and progressive_bottom_ctc_decoder_ablation_bars.pdf before finalizing."
  )
  if [ "${CALLBACK_DRY_RUN:-0}" = "1" ]; then
    callback_args+=(--dry-run)
  fi
  if [ "${CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
    callback_args+=(--check-only)
  fi

  python3 "${callback_args[@]}"
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${callback_status}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

set -euo pipefail

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULTS_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-51 progressive_bottom_ctc_decoder eval starting"
echo "repo=${REPO_ROOT}"
echo "branch=${GIT_BRANCH}"
echo "commit=${GIT_COMMIT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "results_path=${RESULTS_PATH}"
echo "log_path=${LOG_PATH}"
echo "callback_script=${CALLBACK_SCRIPT}"

if [ "${CALLBACK_SMOKE_ONLY:-0}" = "1" ]; then
  echo "CALLBACK_SMOKE_ONLY=1; exiting before GPU eval to exercise the EXIT trap callback path."
  exit 0
fi

if [ ! -f paths.yaml ]; then
  echo "Missing ${REPO_ROOT}/paths.yaml; copy paths_template.yaml and fill dataset/checkpoint paths before launching." >&2
  exit 2
fi

cd lcasr
rm -f "${RESULTS_PATH}"/*.pkl

DATASET=earnings22 \
SPLIT=test \
EPOCH=1 \
SEQ=16384 \
OVERLAP=14336 \
REPEATS=1 \
PYTHON_BIN="${PYTHON_BIN}" \
LRS="9e-6 9e-5 9e-4" \
FAMILIES="progressive_bottom_ctc_decoder" \
RESULTS_ROOT="${RESULTS_ROOT}" \
bash launch_scripts/tune_ctc_self_training_extra_ablation_sweeps.sh

DATASET=tedlium \
SPLIT=test \
EPOCH=1 \
SEQ=16384 \
OVERLAP=14336 \
REPEATS=1 \
PYTHON_BIN="${PYTHON_BIN}" \
LRS="9e-5" \
FAMILIES="progressive_bottom_ctc_decoder" \
RESULTS_ROOT="${RESULTS_ROOT}" \
bash launch_scripts/tune_ctc_self_training_extra_ablation_sweeps.sh

cd results/ctc_self_training_extra_ablation_sweeps
"${PYTHON_BIN}" aggregate.py
"${PYTHON_BIN}" plot_ablation_sweeps.py --refresh --groups progressive_bottom_ctc_decoder

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-51 progressive_bottom_ctc_decoder eval completed"
