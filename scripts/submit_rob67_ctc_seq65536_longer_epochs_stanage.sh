#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"/mnt/parscratch/users/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-67"}
ARTIFACT_DIR=${ARTIFACT_DIR:-"/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-67"}
DATASETS=${DATASETS:-"earnings22 tedlium chime6 rev16"}
EPOCHS=${EPOCHS:-"10 20"}
LRS=${LRS:-"9e-5 3e-4"}
ADAPTED_RESULTS=${ADAPTED_RESULTS:-"${REPO_ROOT}/lcasr/results/ctc_seq65536_self_training_longer_epochs"}
QUEUED_COMMAND=${QUEUED_COMMAND:-"bash scripts/submit_rob67_ctc_seq65536_longer_epochs_stanage.sh"}
CALLBACK_NOTE=${CALLBACK_NOTE:-"ROB-67 follow-up requested after PR #18 result review: 10 and 20 adaptation epochs at LRs 9e-5 and 3e-4 for the 65536-context CTC adapted eval. Expected outputs: 16 adapted PKLs for 4 datasets x epochs {10,20} x LRs {9e-5,3e-4}, plus refreshed summary tables."}

mkdir -p "$ARTIFACT_DIR"
cd "$REPO_ROOT"

read -r -a DATASETS_ARR <<< "$DATASETS"
read -r -a EPOCHS_ARR <<< "$EPOCHS"
read -r -a LRS_ARR <<< "$LRS"
expected_count=$((${#DATASETS_ARR[@]} * ${#EPOCHS_ARR[@]} * ${#LRS_ARR[@]}))

echo "repo_root=${REPO_ROOT}"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"
echo "datasets=${DATASETS}"
echo "epochs=${EPOCHS}"
echo "lrs=${LRS}"
echo "adapted_results=${ADAPTED_RESULTS}"
echo "expected_count=${expected_count}"

array_job_id="$(
  sbatch --parsable \
    --array="0-$((expected_count - 1))" \
    --output="${ARTIFACT_DIR}/longer-epochs-array-%A_%a.out" \
    --export=ALL,DATASETS="${DATASETS}",EPOCHS="${EPOCHS}",LRS="${LRS}",ADAPTED_RESULTS_DIR="${ADAPTED_RESULTS}" \
    scripts/run_rob67_ctc_seq65536_higher_lr_array.sbatch
)"
finalizer_job_id="$(
  sbatch --parsable \
    --dependency="afterany:${array_job_id}" \
    --output="${ARTIFACT_DIR}/longer-epochs-finalize-%j.out" \
    --export=ALL,ARRAY_JOB_ID="${array_job_id}",ADAPTED_RESULTS="${ADAPTED_RESULTS}",EXPECTED_ADAPTED_COUNT="${expected_count}",LOG_PREFIX="longer-epochs-finalize",QUEUED_COMMAND="${QUEUED_COMMAND}",CALLBACK_NOTE="${CALLBACK_NOTE}" \
    scripts/run_rob67_ctc_seq65536_higher_lr_finalize.sbatch
)"

cat <<EOF
array_job_id=${array_job_id}
finalizer_job_id=${finalizer_job_id}
status_command=squeue -j ${array_job_id},${finalizer_job_id} -o '%i|%j|%T|%R|%S|%M|%l|%P'
completion_command=sacct -j ${array_job_id},${finalizer_job_id} --format=JobID,JobName%28,State,ExitCode,Elapsed
artifact_dir=${ARTIFACT_DIR}
results_dir=${ADAPTED_RESULTS}
EOF
