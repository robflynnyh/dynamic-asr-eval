#!/usr/bin/env bash
#SBATCH --job-name=rob67-cpu-smoke
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-67/cpu-smoke-%j.out

set -uo pipefail

REPO_ROOT=${REPO_ROOT:-"/mnt/parscratch/users/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-67"}
LINEAR_ENV_FILE=${LINEAR_ENV_FILE:-"$HOME/.config/dynamic-asr-eval/linear.env"}
LOG_PATH=${LOG_PATH:-"/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-67/cpu-smoke-${SLURM_JOB_ID:-manual}.log"}
RESULTS_PATH=${RESULTS_PATH:-"${REPO_ROOT}/lcasr/results/ctc_seq65536_cpu_smoke"}
PYTHON_BIN=${PYTHON_BIN:-"/mnt/parscratch/users/acp21rjf/conda/main/bin/python"}
CHECKPOINT=${CHECKPOINT:-"/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_65536_rp_1/step_105360.pt"}
CALLBACK_DRY_RUN=${CALLBACK_DRY_RUN:-1}
CALLBACK_TARGET_STATE=${CALLBACK_TARGET_STATE:-Todo}
QUEUED_COMMAND=${QUEUED_COMMAND:-"sbatch scripts/run_rob67_ctc_seq65536_cpu_smoke.sbatch"}
CALLBACK_NOTE=${CALLBACK_NOTE:-"ROB-67 CPU smoke for the 65536-context CTC eval path. This exercises the same cell runner on Stanage without a GPU allocation and is dry-run callback-only by default."}

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
    --results "$RESULTS_PATH"
    --runner-label "slurm:${SLURM_JOB_ID:-manual}:cpu-smoke"
    --queued-command "$QUEUED_COMMAND"
    --branch "$(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || printf unknown)"
    --commit "$(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || printf unknown)"
    --target-state "$CALLBACK_TARGET_STATE"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-12000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-50000}"
    --note "$CALLBACK_NOTE"
  )
  if [ "$CALLBACK_DRY_RUN" = "1" ]; then
    callback_args+=(--dry-run)
  fi
  python3 "${REPO_ROOT}/scripts/linear_experiment_callback.py" "${callback_args[@]}" >> "$LOG_PATH" 2>&1
  exit "$status"
}
trap on_exit EXIT

set -euo pipefail
mkdir -p "$(dirname "$LOG_PATH")" "$RESULTS_PATH"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-67 CPU smoke starting"
echo "repo_root=${REPO_ROOT}"
echo "python_bin=${PYTHON_BIN}"
echo "checkpoint=${CHECKPOINT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

ROB67_WRITE_STANAGE_PATHS=1 \
REPO_ROOT="$REPO_ROOT" \
MODE=unadapted \
DATASET=tedlium \
MAX_RECORDS=1 \
PYTHON_BIN="$PYTHON_BIN" \
CHECKPOINT="$CHECKPOINT" \
UNADAPTED_RESULTS_DIR="$RESULTS_PATH" \
bash "${REPO_ROOT}/scripts/run_rob67_ctc_seq65536_cell.sh"
