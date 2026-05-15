#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"/exp/exp4/acp21rjf/symphony-workspaces-dynamic-asr-eval/ROB-67"}
MODE=${MODE:?MODE must be adapted or unadapted}
DATASET=${DATASET:?DATASET is required}
EPOCH=${EPOCH:-0}
LR=${LR:-1e-5}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
CHECKPOINT=${CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_65536_rp_1/step_105360.pt"}
SEQ=${SEQ:-65536}
OVERLAP=${OVERLAP:-57344}
SPLIT=${SPLIT:-test}
REPEATS=${REPEATS:-1}
MAX_RECORDS=${MAX_RECORDS:-}
UNADAPTED_RESULTS_DIR=${UNADAPTED_RESULTS_DIR:-"./results/ctc_seq65536_unadapted_baseline"}
ADAPTED_RESULTS_DIR=${ADAPTED_RESULTS_DIR:-"./results/ctc_seq65536_self_training_eval"}

write_stanage_paths_yaml() {
  cat > "${REPO_ROOT}/paths.yaml" <<'YAML'
datasets:
  tedlium:
    test: /mnt/parscratch/users/acp21rjf/TEDLIUM_release1/test
    dev: /mnt/parscratch/users/acp21rjf/TEDLIUM_release1/dev
    train: /mnt/parscratch/users/acp21rjf/TEDLIUM_release1/train
  earnings:
    test: /mnt/parscratch/users/acp21rjf/earnings22/test_original
    dev: /mnt/parscratch/users/acp21rjf/earnings22/dev_original
    text: /mnt/parscratch/users/acp21rjf/earnings22/full_transcripts.json
  chime6:
    audio:
      test: /mnt/parscratch/users/acp21rjf/chime6/audio/eval
      dev: /mnt/parscratch/users/acp21rjf/chime6/audio/dev
    text:
      test: /mnt/parscratch/users/acp21rjf/chime6/transcriptions/eval
      dev: /mnt/parscratch/users/acp21rjf/chime6/transcriptions/dev
  rev16:
    test: /mnt/parscratch/users/acp21rjf/rev_benchmark
checkpoints:
  lcasr: /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_65536_rp_1/step_105360.pt
  nvidia_ctc: nvidia/stt_en_fastconformer_ctc_large
  wav2vec2: facebook/wav2vec2-large-960h-lv60-self
  lm: /mnt/parscratch/users/acp21rjf/spotify/512_1280/step_540012.pt
YAML
}

if [ "${ROB67_WRITE_STANAGE_PATHS:-0}" = "1" ]; then
  write_stanage_paths_yaml
fi

cd "$REPO_ROOT"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-67 cell starting"
echo "repo_root=${REPO_ROOT}"
echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf unknown)"
echo "commit=$(git rev-parse HEAD 2>/dev/null || printf unknown)"
echo "mode=${MODE}"
echo "dataset=${DATASET}"
echo "epoch=${EPOCH}"
echo "lr=${LR}"
echo "checkpoint=${CHECKPOINT}"
echo "python_bin=${PYTHON_BIN}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "slurm_job_id=${SLURM_JOB_ID:-unset}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-unset}"

cd "$REPO_ROOT/lcasr"

COMMON_ENV=(
  PYTHON_BIN="$PYTHON_BIN"
  CHECKPOINT="$CHECKPOINT"
  DATASETS="$DATASET"
  REPEATS="$REPEATS"
  SEQ="$SEQ"
  OVERLAP="$OVERLAP"
  SPLIT="$SPLIT"
)

if [ -n "$MAX_RECORDS" ]; then
  COMMON_ENV+=(MAX_RECORDS="$MAX_RECORDS")
fi

case "$MODE" in
  unadapted)
    env "${COMMON_ENV[@]}" \
      RESULTS_DIR="$UNADAPTED_RESULTS_DIR" \
      bash launch_scripts/run_ctc_seq65536_unadapted_baseline.sh
    ;;
  adapted)
    env "${COMMON_ENV[@]}" \
      RESULTS_DIR="$ADAPTED_RESULTS_DIR" \
      EPOCHS="$EPOCH" \
      LR="$LR" \
      LRS="$LR" \
      bash launch_scripts/run_ctc_seq65536_self_training_eval.sh
    ;;
  *)
    echo "Unknown MODE=${MODE}; expected adapted or unadapted" >&2
    exit 2
    ;;
esac
