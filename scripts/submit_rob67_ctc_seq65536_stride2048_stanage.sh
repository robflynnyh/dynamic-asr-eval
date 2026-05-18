#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"/mnt/parscratch/users/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-67"}
ARTIFACT_DIR=${ARTIFACT_DIR:-"/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-67"}
DATASETS=${DATASETS:-"earnings22 tedlium chime6 rev16"}
EPOCHS=${EPOCHS:-"5"}
LRS=${LRS:-"9e-5"}
SEQ=${SEQ:-65536}
OVERLAP=${OVERLAP:-63488}
ADAPTED_RESULTS=${ADAPTED_RESULTS:-"${REPO_ROOT}/lcasr/results/ctc_seq65536_self_training_stride2048"}

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
echo "seq=${SEQ}"
echo "overlap=${OVERLAP}"
echo "stride=$((SEQ - OVERLAP))"
echo "adapted_results=${ADAPTED_RESULTS}"
echo "expected_count=${expected_count}"

array_job_id="$(
  sbatch --parsable \
    --array="0-$((expected_count - 1))" \
    --output="${ARTIFACT_DIR}/stride2048-array-%A_%a.out" \
    --export=ALL,DATASETS="${DATASETS}",EPOCHS="${EPOCHS}",LRS="${LRS}",SEQ="${SEQ}",OVERLAP="${OVERLAP}",ADAPTED_RESULTS_DIR="${ADAPTED_RESULTS}" \
    scripts/run_rob67_ctc_seq65536_higher_lr_array.sbatch
)"
finalizer_job_id="$(
  sbatch --parsable \
    --dependency="afterany:${array_job_id}" \
    --output="${ARTIFACT_DIR}/stride2048-finalize-%j.out" \
    --export=ALL,ARRAY_JOB_ID="${array_job_id}",ADAPTED_RESULTS="${ADAPTED_RESULTS}",EXPECTED_ADAPTED_COUNT="${expected_count}",LOG_PREFIX="stride2048-finalize",FINALIZER_KIND="stride2048" \
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
