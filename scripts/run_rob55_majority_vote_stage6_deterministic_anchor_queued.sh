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
SCREEN_NAME="${SCREEN_NAME:-rob55_majority_vote_stage6_deterministic_anchor}"
RESULTS_PATH="${RESULTS_PATH:-lcasr/results/enc_dec/enc_dec_majority_vote_stage6_deterministic_anchor}"
LOG_PATH="${LOG_PATH:-${RESULTS_PATH}/logs/rob55_majority_vote_stage6_deterministic_anchor.log}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob55_majority_vote_stage6_deterministic_anchor_queued.sh}"
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
    --note "ROB-55 Stage 6 deterministic-anchored vote run finished. Inspect summary.csv to see whether using stochastic votes as a confidence gate for deterministic beam labels helps CE, GRPO, or MAXRL." \
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
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-55 Stage 6 deterministic-anchored vote sweep"
  echo "repo=${REPO_ROOT}"
  echo "branch=${GIT_BRANCH}"
  echo "commit=${GIT_COMMIT}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "queued_command=${QUEUED_COMMAND}"
  echo "results=${RESULTS_PATH}"
  echo "vote_sample_counts=${VOTE_SAMPLE_COUNTS:-8 16}"
} | tee "$LOG_PATH"

if [ "${EXPERIMENT_DRY_RUN:-0}" = "1" ]; then
  echo "EXPERIMENT_DRY_RUN=1; skipping Stage 6 payload" | tee -a "$LOG_PATH"
  exit 0
fi

cd lcasr
first_group=1
for vote_samples in ${VOTE_SAMPLE_COUNTS:-8 16}; do
  for group in ce rl; do
    if [ "$group" = "ce" ]; then
      training_modes="teacher_ce"
      lrs="1e-7 3e-7"
    else
      training_modes="grpo maxrl"
      lrs="1e-7"
    fi

    run_baseline=0
    if [ "$first_group" = "1" ]; then
      run_baseline=1
      first_group=0
    fi

    GPU="${CUDA_VISIBLE_DEVICES:-0}" \
    RESULTS_DIR="./results/enc_dec/enc_dec_majority_vote_stage6_deterministic_anchor" \
    RUN_BASELINE="$run_baseline" \
    REPEATS=1 \
    TRAINING_MODES="$training_modes" \
    LRS="$lrs" \
    AUGS="freq3_width24_time0" \
    VOTE_TEMPS="0.7 1.0" \
    VOTE_MIN_COUNTS="2" \
    VOTE_SIMILARITIES="1.0 0.95" \
    TEACHER_VOTE_NUM_SAMPLES="$vote_samples" \
    TEACHER_VOTE_INCLUDE_DETERMINISTIC=1 \
    TEACHER_VOTE_REPRESENTATIVE_STRATEGY="deterministic" \
    DRY_RUN="${DRY_RUN:-0}" \
    bash launch_scripts/tune_enc_dec_majority_vote_tedlium_dev.sh 2>&1 | tee -a "../${LOG_PATH}"
  done
done

python results/enc_dec/enc_dec_majority_vote/aggregate.py \
  --directory results/enc_dec/enc_dec_majority_vote_stage6_deterministic_anchor \
  --csv results/enc_dec/enc_dec_majority_vote_stage6_deterministic_anchor/summary.csv \
  2>&1 | tee -a "../${LOG_PATH}"
