#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"/mnt/parscratch/users/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-67"}
ARTIFACT_DIR=${ARTIFACT_DIR:-"/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-67"}

mkdir -p "$ARTIFACT_DIR"
cd "$REPO_ROOT"

echo "repo_root=${REPO_ROOT}"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"

array_job_id="$(
  sbatch --parsable scripts/run_rob67_ctc_seq65536_stanage_array.sbatch
)"
finalizer_job_id="$(
  sbatch --parsable \
    --dependency="afterany:${array_job_id}" \
    --export=ALL,ARRAY_JOB_ID="${array_job_id}" \
    scripts/run_rob67_ctc_seq65536_stanage_finalize.sbatch
)"

cat <<EOF
array_job_id=${array_job_id}
finalizer_job_id=${finalizer_job_id}
status_command=squeue -j ${array_job_id},${finalizer_job_id} -o '%i|%j|%T|%R|%S|%M|%l|%P'
completion_command=sacct -j ${array_job_id},${finalizer_job_id} --format=JobID,JobName%24,State,ExitCode,Elapsed
artifact_dir=${ARTIFACT_DIR}
EOF
