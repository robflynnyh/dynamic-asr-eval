#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-122}"
SCREEN_NAME="${SCREEN_NAME:-rob122_rob81_floras_eval}"
RESULTS_PATH="${RESULTS_PATH:-lcasr/results/enc_dec/rob81_floras50_finetune_eval}"
LOG_PATH="${LOG_PATH:-${RESULTS_PATH}/screen.log}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob122_rob81_floras_eval_queued.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

SCRATCH_ROOT="${SCRATCH_ROOT:-symphony/.scratch/ROB-122}"
mkdir -p \
  "$SCRATCH_ROOT/tmp" \
  "$SCRATCH_ROOT/matplotlib" \
  "$SCRATCH_ROOT/hf_home" \
  "$SCRATCH_ROOT/xdg_cache" \
  "$RESULTS_PATH" \
  "$(dirname "$LOG_PATH")"

export TMPDIR="${TMPDIR:-$PWD/$SCRATCH_ROOT/tmp}"
export TEMP="${TEMP:-$TMPDIR}"
export TMP="${TMP:-$TMPDIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/$SCRATCH_ROOT/matplotlib}"
export HF_HOME="${HF_HOME:-$PWD/$SCRATCH_ROOT/hf_home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/$SCRATCH_ROOT/xdg_cache}"

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
    --max-log-chars 20000 \
    --max-body-chars 60000 \
    --note "ROB-122 ROB-81 Floras-finetuned encoder-decoder eval wrapper completed. Inspect \`lcasr/results/enc_dec/rob81_floras50_finetune_eval/OUTCOME.md\`, \`summary.csv\`, PKLs, and logs before finalizing. Rev16 dev is expected to remain unavailable because the current Rev16 loader exposes test only." \
    "${callback_args[@]}"
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${callback_status}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

if [ "${ROB122_SMOKE:-0}" = "1" ]; then
  {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-122 wrapper smoke"
    echo "branch=${GIT_BRANCH}"
    echo "commit=${GIT_COMMIT}"
    echo "results_path=${RESULTS_PATH}"
    echo "tmpdir=${TMPDIR}"
    echo "mplconfigdir=${MPLCONFIGDIR}"
  } 2>&1 | tee -a "$LOG_PATH"
  exit 0
fi

GPU="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"
export GPU

{
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-122 ROB-81 Floras-finetuned encoder-decoder eval"
  echo "branch=${GIT_BRANCH}"
  echo "commit=${GIT_COMMIT}"
  echo "gpu=${GPU}"
  echo "checkpoint=/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0/step_323484.pt"
  echo "rows=tedlium/dev tedlium/test earnings22/dev earnings22/test chime6/dev chime6/test rev16/test"
  echo "baseline=no_adapt beam5_lp0p5 epoch0"
  echo "adaptation=teacher_ce beam5_lp0p5 epoch1 lr1e-7 freq3_width24_time0"
  echo "seq=2048 overlap=0 repeats=1"
  echo "results_path=${RESULTS_PATH}"
  echo "tmpdir=${TMPDIR}"
} 2>&1 | tee -a "$LOG_PATH"

RESULTS_ROOT=./results/enc_dec/rob81_floras50_finetune_eval \
SKIP_EXISTING="${SKIP_EXISTING:-1}" \
bash lcasr/launch_scripts/run_rob122_rob81_floras_eval.sh 2>&1 | tee -a "$LOG_PATH"
