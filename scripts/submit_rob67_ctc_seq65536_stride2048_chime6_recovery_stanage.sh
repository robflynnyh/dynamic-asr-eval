#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"/mnt/parscratch/users/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-67"}
ARTIFACT_DIR=${ARTIFACT_DIR:-"/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-67"}
DATASETS=${DATASETS:-"chime6"}
EPOCHS=${EPOCHS:-"5"}
LRS=${LRS:-"9e-5"}
SEQ=${SEQ:-65536}
OVERLAP=${OVERLAP:-63488}
RECOVERY_MEM=${RECOVERY_MEM:-80G}
EXPECTED_ADAPTED_COUNT=${EXPECTED_ADAPTED_COUNT:-4}
ADAPTED_RESULTS=${ADAPTED_RESULTS:-"${REPO_ROOT}/lcasr/results/seq_65536_investigation/self_training_stride2048"}

mkdir -p "$ARTIFACT_DIR"
cd "$REPO_ROOT"

read -r -a DATASETS_ARR <<< "$DATASETS"
read -r -a EPOCHS_ARR <<< "$EPOCHS"
read -r -a LRS_ARR <<< "$LRS"
recovery_count=$((${#DATASETS_ARR[@]} * ${#EPOCHS_ARR[@]} * ${#LRS_ARR[@]}))

echo "repo_root=${REPO_ROOT}"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"
echo "datasets=${DATASETS}"
echo "epochs=${EPOCHS}"
echo "lrs=${LRS}"
echo "seq=${SEQ}"
echo "overlap=${OVERLAP}"
echo "stride=$((SEQ - OVERLAP))"
echo "recovery_mem=${RECOVERY_MEM}"
echo "adapted_results=${ADAPTED_RESULTS}"
echo "recovery_count=${recovery_count}"
echo "expected_adapted_count_after_recovery=${EXPECTED_ADAPTED_COUNT}"

array_job_id="$(
  sbatch --parsable \
    --array="0-$((recovery_count - 1))" \
    --mem="${RECOVERY_MEM}" \
    --output="${ARTIFACT_DIR}/stride2048-chime6-recovery-array-%A_%a.out" \
    --export=ALL,DATASETS="${DATASETS}",EPOCHS="${EPOCHS}",LRS="${LRS}",SEQ="${SEQ}",OVERLAP="${OVERLAP}",ADAPTED_RESULTS_DIR="${ADAPTED_RESULTS}" \
    scripts/run_rob67_ctc_seq65536_higher_lr_array.sbatch
)"
finalizer_job_id="$(
  sbatch --parsable \
    --dependency="afterany:${array_job_id}" \
    --output="${ARTIFACT_DIR}/stride2048-chime6-recovery-finalize-%j.out" \
    --export=ALL,ARRAY_JOB_ID="${array_job_id}",ADAPTED_RESULTS="${ADAPTED_RESULTS}",EXPECTED_ADAPTED_COUNT="${EXPECTED_ADAPTED_COUNT}",LOG_PREFIX="stride2048-chime6-recovery-finalize",FINALIZER_KIND="stride2048-chime6-recovery" \
    scripts/run_rob67_ctc_seq65536_higher_lr_finalize.sbatch
)"

cat <<EOF
array_job_id=${array_job_id}
finalizer_job_id=${finalizer_job_id}
status_command=squeue -j ${array_job_id},${finalizer_job_id} -o '%i|%j|%T|%R|%S|%M|%l|%P'
completion_command=sacct -j ${array_job_id},${finalizer_job_id} --format=JobID,JobName%28,State,ExitCode,Elapsed,MaxRSS,ReqMem
artifact_dir=${ARTIFACT_DIR}
results_dir=${ADAPTED_RESULTS}
EOF
