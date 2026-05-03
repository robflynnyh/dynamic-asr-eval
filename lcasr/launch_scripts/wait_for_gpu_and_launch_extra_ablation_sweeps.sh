#!/usr/bin/env bash
set -euo pipefail

# Wait until a selected GPU has no compute processes, then launch the extra
# CTC self-training ablation sweeps on that GPU.
#
# Example:
#   GPU=2 bash launch_scripts/wait_for_gpu_and_launch_extra_ablation_sweeps.sh
#
# Optional controls:
#   CHECK_INTERVAL_SECONDS=60
#   LAUNCH_CMD='bash launch_scripts/tune_ctc_self_training_extra_ablation_sweeps.sh'

GPU=${GPU:-2}
CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS:-60}
LAUNCH_CMD=${LAUNCH_CMD:-"bash launch_scripts/tune_ctc_self_training_extra_ablation_sweeps.sh"}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found" >&2
    exit 1
fi

GPU_UUID=$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')
if [ -z "$GPU_UUID" ]; then
    echo "Could not resolve GPU ${GPU} to a UUID" >&2
    exit 1
fi

echo "Waiting for GPU ${GPU} (${GPU_UUID}) to have no compute processes..."
echo "Check interval: ${CHECK_INTERVAL_SECONDS}s"
echo "Launch command: GPU=${GPU} ${LAUNCH_CMD}"

while true
do
    BUSY_PIDS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' -v uuid="$GPU_UUID" '$1 == uuid {print $2}' \
        | tr '\n' ' ' \
        | sed 's/[[:space:]]*$//')

    if [ -z "$BUSY_PIDS" ]; then
        echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] GPU ${GPU} is free; launching."
        GPU="$GPU" bash -lc "$LAUNCH_CMD"
        exit $?
    fi

    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] GPU ${GPU} busy with pid(s): ${BUSY_PIDS}; waiting ${CHECK_INTERVAL_SECONDS}s."
    sleep "$CHECK_INTERVAL_SECONDS"
done
