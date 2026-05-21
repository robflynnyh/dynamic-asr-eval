#!/usr/bin/env bash
#SBATCH --job-name=rob67-seq65536-hilr
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-15
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-67/higher-lr-array-%A_%a.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"/mnt/parscratch/users/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-67"}
PYTHON_BIN=${PYTHON_BIN:-"/mnt/parscratch/users/acp21rjf/conda/main/bin/python"}
CHECKPOINT=${CHECKPOINT:-"/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_65536_rp_1/step_105360.pt"}
ADAPTED_RESULTS_DIR=${ADAPTED_RESULTS_DIR:-"./results/seq_65536_investigation/self_training_higher_lr"}
DATASETS_STR=${DATASETS:-"earnings22 tedlium chime6 rev16"}
EPOCHS_STR=${EPOCHS:-"1 5"}
LRS_STR=${LRS:-"9e-5 3e-4"}

read -r -a DATASETS_ARR <<< "$DATASETS_STR"
read -r -a EPOCHS_ARR <<< "$EPOCHS_STR"
read -r -a LRS_ARR <<< "$LRS_STR"

dataset_count=${#DATASETS_ARR[@]}
epoch_count=${#EPOCHS_ARR[@]}
lr_count=${#LRS_ARR[@]}
expected_count=$((dataset_count * epoch_count * lr_count))
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

if [ "$task_id" -lt 0 ] || [ "$task_id" -ge "$expected_count" ]; then
  echo "Unexpected task id ${task_id}; expected 0-$((expected_count - 1))" >&2
  exit 2
fi

dataset_idx=$((task_id % dataset_count))
epoch_idx=$(((task_id / dataset_count) % epoch_count))
lr_idx=$((task_id / (dataset_count * epoch_count)))

ROB67_WRITE_STANAGE_PATHS="${ROB67_WRITE_STANAGE_PATHS:-1}" \
REPO_ROOT="$REPO_ROOT" \
MODE=adapted \
DATASET="${DATASETS_ARR[$dataset_idx]}" \
EPOCH="${EPOCHS_ARR[$epoch_idx]}" \
LR="${LRS_ARR[$lr_idx]}" \
PYTHON_BIN="$PYTHON_BIN" \
CHECKPOINT="$CHECKPOINT" \
ADAPTED_RESULTS_DIR="$ADAPTED_RESULTS_DIR" \
bash "${REPO_ROOT}/scripts/run_rob67_ctc_seq65536_cell.sh"
