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
SCREEN_NAME="${SCREEN_NAME:-rob55_majority_vote_stage2_medoid_confidence}"
RESULTS_PATH="${RESULTS_PATH:-lcasr/results/enc_dec/enc_dec_majority_vote_stage2_medoid_confidence}"
LOG_PATH="${LOG_PATH:-${RESULTS_PATH}/logs/rob55_majority_vote_stage2_medoid_confidence.log}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob55_majority_vote_stage2_medoid_confidence_queued.sh}"
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
    --note "ROB-55 Stage 2 medoid/confidence majority-vote sweep finished. Inspect summary.csv and vote-retention log counts before deciding whether to expand to Earnings22 or test-set repeats." \
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
{
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-55 majority-vote Stage 2 medoid/confidence sweep"
  echo "repo=${REPO_ROOT}"
  echo "branch=${GIT_BRANCH}"
  echo "commit=${GIT_COMMIT}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "queued_command=${QUEUED_COMMAND}"
  echo "results=${RESULTS_PATH}"
} | tee "$LOG_PATH"

cd lcasr
GPU="${CUDA_VISIBLE_DEVICES:-0}" \
RESULTS_DIR="./results/enc_dec/enc_dec_majority_vote_stage2_medoid_confidence" \
RUN_BASELINE=1 \
REPEATS=2 \
TRAINING_MODES="teacher_ce teacher_kl" \
LRS="1e-7 3e-7" \
AUGS="freq3_width24_time0" \
VOTE_TEMPS="0.7" \
VOTE_MIN_COUNTS="3" \
VOTE_SIMILARITIES="0.9 0.95" \
TEACHER_VOTE_REPRESENTATIVE_STRATEGY="medoid" \
TEACHER_FILTER_LOW_CONFIDENCE=1 \
TEACHER_MIN_MEAN_MAX_PROB=0.35 \
TEACHER_MAX_MEAN_ENTROPY=2.5 \
DRY_RUN="${EXPERIMENT_DRY_RUN:-${DRY_RUN:-0}}" \
bash launch_scripts/tune_enc_dec_majority_vote_tedlium_dev.sh 2>&1 | tee -a "../${LOG_PATH}"

python results/enc_dec/enc_dec_majority_vote/aggregate.py \
  --directory results/enc_dec/enc_dec_majority_vote_stage2_medoid_confidence \
  --csv results/enc_dec/enc_dec_majority_vote_stage2_medoid_confidence/summary.csv \
  2>&1 | tee -a "../${LOG_PATH}"
