#!/usr/bin/env bash
#SBATCH --job-name=rob67-seq65536
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-67/array-%A_%a.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"/mnt/parscratch/users/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-67"}
PYTHON_BIN=${PYTHON_BIN:-"/mnt/parscratch/users/acp21rjf/conda/main/bin/python"}
CHECKPOINT=${CHECKPOINT:-"/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_65536_rp_1/step_105360.pt"}
LR=${LR:-1e-5}

case "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" in
  0) MODE=unadapted; DATASET=earnings22; EPOCH=0 ;;
  1) MODE=unadapted; DATASET=tedlium; EPOCH=0 ;;
  2) MODE=unadapted; DATASET=chime6; EPOCH=0 ;;
  3) MODE=unadapted; DATASET=rev16; EPOCH=0 ;;
  4) MODE=adapted; DATASET=earnings22; EPOCH=1 ;;
  5) MODE=adapted; DATASET=tedlium; EPOCH=1 ;;
  6) MODE=adapted; DATASET=chime6; EPOCH=1 ;;
  7) MODE=adapted; DATASET=rev16; EPOCH=1 ;;
  8) MODE=adapted; DATASET=earnings22; EPOCH=5 ;;
  9) MODE=adapted; DATASET=tedlium; EPOCH=5 ;;
  10) MODE=adapted; DATASET=chime6; EPOCH=5 ;;
  11) MODE=adapted; DATASET=rev16; EPOCH=5 ;;
  *) echo "Unexpected task id: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

ROB67_WRITE_STANAGE_PATHS=1 \
REPO_ROOT="$REPO_ROOT" \
MODE="$MODE" \
DATASET="$DATASET" \
EPOCH="$EPOCH" \
LR="$LR" \
PYTHON_BIN="$PYTHON_BIN" \
CHECKPOINT="$CHECKPOINT" \
bash "${REPO_ROOT}/scripts/run_rob67_ctc_seq65536_cell.sh"
