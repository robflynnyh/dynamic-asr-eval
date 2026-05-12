#!/usr/bin/env bash
#SBATCH --job-name=rob67-hilr-finalize
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-67/higher-lr-finalize-%j.out

set -uo pipefail

REPO_ROOT=${REPO_ROOT:-"/mnt/parscratch/users/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-67"}
LINEAR_ENV_FILE=${LINEAR_ENV_FILE:-"$HOME/.config/dynamic-asr-eval/linear.env"}
ARRAY_JOB_ID=${ARRAY_JOB_ID:-unknown}
LOG_PATH=${LOG_PATH:-"/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-67/higher-lr-finalize-${SLURM_JOB_ID:-manual}.log"}
ADAPTED_RESULTS=${ADAPTED_RESULTS:-"${REPO_ROOT}/lcasr/results/ctc_seq65536_self_training_higher_lr"}
CALLBACK_TARGET_STATE=${CALLBACK_TARGET_STATE:-Todo}
PYTHON_BIN=${PYTHON_BIN:-"/mnt/parscratch/users/acp21rjf/conda/main/bin/python"}
EXPECTED_ADAPTED_COUNT=${EXPECTED_ADAPTED_COUNT:-16}

on_exit() {
  status=$?
  set +e
  if [ -f "$LINEAR_ENV_FILE" ]; then
    set -a
    . "$LINEAR_ENV_FILE"
    set +a
  fi
  callback_args=(
    --issue ROB-67
    --status-code "$status"
    --log "$LOG_PATH"
    --results "$ADAPTED_RESULTS"
    --runner-label "slurm-array:${ARRAY_JOB_ID};higher-lr-finalizer:${SLURM_JOB_ID:-manual}"
    --queued-command "bash scripts/submit_rob67_ctc_seq65536_higher_lr_stanage.sh"
    --branch "$(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || printf unknown)"
    --commit "$(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || printf unknown)"
    --target-state "$CALLBACK_TARGET_STATE"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-20000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-60000}"
    --note "ROB-67 higher-LR follow-up for the completed 65536-context CTC adapted eval. Expected outputs: 16 adapted PKLs for 4 datasets x 2 epoch counts x LRs {9e-5,3e-4}, plus refreshed summary tables."
  )
  if [ "${CALLBACK_DRY_RUN:-0}" = "1" ]; then
    callback_args+=(--dry-run)
  fi
  if [ -z "${LINEAR_API_KEY:-}" ] && [ "${CALLBACK_DRY_RUN:-0}" != "1" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "$status"
  fi
  python3 "${REPO_ROOT}/scripts/linear_experiment_callback.py" "${callback_args[@]}" >> "$LOG_PATH" 2>&1
  exit "$status"
}
trap on_exit EXIT

set -euo pipefail
mkdir -p "$(dirname "$LOG_PATH")" "$ADAPTED_RESULTS"
exec > >(tee -a "$LOG_PATH") 2>&1

cd "$REPO_ROOT"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-67 higher-LR finalizer starting"
echo "repo_root=${REPO_ROOT}"
echo "array_job_id=${ARRAY_JOB_ID}"
echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf unknown)"
echo "commit=$(git rev-parse HEAD 2>/dev/null || printf unknown)"
echo "adapted_results=${ADAPTED_RESULTS}"

"$PYTHON_BIN" lcasr/results/ctc_seq65536_self_training_eval/aggregate.py --root "$ADAPTED_RESULTS"

adapted_count=$(find "$ADAPTED_RESULTS" -maxdepth 1 -name '*-epoch-*-lr-*.pkl' | wc -l)
echo "adapted_count=${adapted_count}/${EXPECTED_ADAPTED_COUNT}"

if command -v sacct >/dev/null 2>&1 && [ "$ARRAY_JOB_ID" != "unknown" ]; then
  sacct -j "$ARRAY_JOB_ID" --format=JobID,JobName%28,State,ExitCode,Elapsed || true
fi

if [ "$adapted_count" -ne "$EXPECTED_ADAPTED_COUNT" ]; then
  echo "Missing expected higher-LR result files; leaving issue for Symphony diagnosis." >&2
  exit 1
fi

echo "ROB-67 higher-LR finalizer found all expected result files."
