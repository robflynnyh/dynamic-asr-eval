#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-55}"
SCREEN_NAME="${SCREEN_NAME:-rob55_segmented_diagnostic}"
RESULTS_PATH="${RESULTS_PATH:-lcasr/results/enc_dec/enc_dec_majority_vote_utterance_diagnostic}"
LOG_PATH="${LOG_PATH:-${RESULTS_PATH}/logs/rob55_segmented_diagnostic.log}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob55_segmented_diagnostic_queued.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
  cd "${REPO_ROOT}"
  if [ -z "${LINEAR_API_KEY:-}" ] && [ "${CALLBACK_DRY_RUN:-0}" != "1" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "${status}"
  fi
  callback_extra_args=()
  if [ "${CALLBACK_DRY_RUN:-0}" = "1" ]; then
    callback_extra_args=(--dry-run)
  fi
  python3 scripts/linear_experiment_callback.py \
    --issue "${LINEAR_ISSUE}" \
    --status-code "${status}" \
    --log "${LOG_PATH}" \
    --results "${RESULTS_PATH}" \
    --screen-name "${SCREEN_NAME}" \
    --runner-label "screen:${SCREEN_NAME}" \
    --queued-command "${QUEUED_COMMAND}" \
    --branch "${GIT_BRANCH}" \
    --commit "${GIT_COMMIT}" \
    --target-state Todo \
    --note "ROB-55 segmented TEDLIUM-dev diagnostic finished. Inspect summary.json/summary.csv for which utterances improved and how teacher-label WER relates to improvement before launching more sweeps." \
    "${callback_extra_args[@]}"
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${callback_status}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

set -euo pipefail

mkdir -p "$(dirname "$LOG_PATH")" "$RESULTS_PATH"
if [[ "$LOG_PATH" = /* ]]; then
  LCASR_LOG_PATH="$LOG_PATH"
elif [[ "$LOG_PATH" == lcasr/* ]]; then
  LCASR_LOG_PATH="${LOG_PATH#lcasr/}"
else
  LCASR_LOG_PATH="../${LOG_PATH}"
fi
if [[ "$RESULTS_PATH" = /* ]]; then
  LCASR_RESULTS_PATH="$RESULTS_PATH"
elif [[ "$RESULTS_PATH" == lcasr/* ]]; then
  LCASR_RESULTS_PATH="${RESULTS_PATH#lcasr/}"
else
  LCASR_RESULTS_PATH="../${RESULTS_PATH}"
fi
{
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-55 segmented TEDLIUM-dev diagnostic"
  echo "repo=${REPO_ROOT}"
  echo "branch=${GIT_BRANCH}"
  echo "commit=${GIT_COMMIT}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "queued_command=${QUEUED_COMMAND}"
  echo "results=${RESULTS_PATH}"
  echo "max_utterances=${MAX_UTTERANCES:-0}"
  echo "max_duration=${MAX_DURATION:-0}"
} | tee "$LOG_PATH"

if [ "${EXPERIMENT_DRY_RUN:-0}" = "1" ]; then
  echo "EXPERIMENT_DRY_RUN=1; skipping segmented diagnostic payload" | tee -a "$LOG_PATH"
  exit 0
fi

cd lcasr
python3.10 results/enc_dec/enc_dec_majority_vote/run_segmented_tedlium_diagnostic.py \
  --output-dir "${LCASR_RESULTS_PATH}" \
  --split dev \
  --checkpoint /store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt \
  --seq-len 2048 \
  --overlap 0 \
  --epochs 1 \
  --training-mode teacher_kl \
  --optim-lr 1e-7 \
  --teacher-kl-temperature 1.0 \
  --teacher-vote-num-samples "${TEACHER_VOTE_NUM_SAMPLES:-8}" \
  --teacher-vote-temperature "${TEACHER_VOTE_TEMPERATURE:-0.7}" \
  --teacher-vote-min-count "${TEACHER_VOTE_MIN_COUNT:-2}" \
  --teacher-vote-similarity "${TEACHER_VOTE_SIMILARITY:-1.0}" \
  --teacher-vote-representative-strategy medoid \
  --enc-dec-beam-width 5 \
  --enc-dec-length-penalty 0.5 \
  --spec-augment-freq-mask-param "${SPEC_AUGMENT_FREQ_MASK_PARAM:-24}" \
  --spec-augment-n-time-masks "${SPEC_AUGMENT_N_TIME_MASKS:-0}" \
  --spec-augment-n-freq-masks "${SPEC_AUGMENT_N_FREQ_MASKS:-3}" \
  --teacher_filter_max_length \
  --teacher_filter_max_consecutive_token_repeat \
  --teacher_filter_repeated_token_ngrams \
  --teacher_repeated_token_ngram_sizes 2 3 \
  --teacher_filter_repeated_words \
  --teacher_filter_ctc_agreement \
  --max-utterances "${MAX_UTTERANCES:-0}" \
  --max-duration "${MAX_DURATION:-0}" \
  2>&1 | tee -a "${LCASR_LOG_PATH}"
