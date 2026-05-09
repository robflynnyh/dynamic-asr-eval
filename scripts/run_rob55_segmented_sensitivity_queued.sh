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
SCREEN_NAME="${SCREEN_NAME:-rob55_segmented_sensitivity}"
RESULTS_ROOT="${RESULTS_ROOT:-lcasr/results/enc_dec/enc_dec_majority_vote_segmented_sensitivity}"
LOG_PATH="${LOG_PATH:-${RESULTS_ROOT}/logs/rob55_segmented_sensitivity.log}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/run_rob55_segmented_sensitivity_queued.sh}"
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
    --results "${RESULTS_ROOT}" \
    --screen-name "${SCREEN_NAME}" \
    --runner-label "screen:${SCREEN_NAME}" \
    --queued-command "${QUEUED_COMMAND}" \
    --branch "${GIT_BRANCH}" \
    --commit "${GIT_COMMIT}" \
    --target-state Todo \
    --note "ROB-55 Stage 5 segmented sensitivity run finished. Inspect per-setting summary.json files to see whether higher LR/epochs, CE vs KL, augmentation, or vote sample count/temperature can make clean teacher samples move utterance-level WER." \
    "${callback_extra_args[@]}"
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${callback_status}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

set -euo pipefail

mkdir -p "$(dirname "$LOG_PATH")" "$RESULTS_ROOT"

if [[ "$RESULTS_ROOT" = /* ]]; then
  LCASR_RESULTS_ROOT="$RESULTS_ROOT"
elif [[ "$RESULTS_ROOT" == lcasr/* ]]; then
  LCASR_RESULTS_ROOT="${RESULTS_ROOT#lcasr/}"
else
  LCASR_RESULTS_ROOT="../${RESULTS_ROOT}"
fi
if [[ "$LOG_PATH" = /* ]]; then
  LCASR_LOG_PATH="$LOG_PATH"
elif [[ "$LOG_PATH" == lcasr/* ]]; then
  LCASR_LOG_PATH="${LOG_PATH#lcasr/}"
else
  LCASR_LOG_PATH="../${LOG_PATH}"
fi

{
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ROB-55 segmented sensitivity sweep"
  echo "repo=${REPO_ROOT}"
  echo "branch=${GIT_BRANCH}"
  echo "commit=${GIT_COMMIT}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "queued_command=${QUEUED_COMMAND}"
  echo "results_root=${RESULTS_ROOT}"
  echo "max_settings=${MAX_SETTINGS:-0}"
  echo "max_utterances=${MAX_UTTERANCES:-0}"
  echo "max_duration=${MAX_DURATION:-0}"
} | tee "$LOG_PATH"

if [ "${EXPERIMENT_DRY_RUN:-0}" = "1" ]; then
  echo "EXPERIMENT_DRY_RUN=1; skipping segmented sensitivity payload" | tee -a "$LOG_PATH"
  exit 0
fi

SETTINGS=(
  "ce_lr1e6_e1_noaug_vote8_t0p7|teacher_ce|1e-6|1|34|0|0|8|0.7|2|1.0"
  "ce_lr1e6_e3_noaug_vote8_t0p7|teacher_ce|1e-6|3|34|0|0|8|0.7|2|1.0"
  "ce_lr3e6_e1_noaug_vote8_t0p7|teacher_ce|3e-6|1|34|0|0|8|0.7|2|1.0"
  "ce_lr1e6_e3_freq3_vote8_t0p7|teacher_ce|1e-6|3|24|3|0|8|0.7|2|1.0"
  "ce_lr1e6_e3_freq6_vote8_t0p7|teacher_ce|1e-6|3|34|6|0|8|0.7|2|1.0"
  "kl_lr1e6_e3_noaug_vote8_t0p7|teacher_kl|1e-6|3|34|0|0|8|0.7|2|1.0"
  "ce_lr1e6_e3_noaug_vote16_t0p7|teacher_ce|1e-6|3|34|0|0|16|0.7|2|1.0"
  "ce_lr1e6_e3_noaug_vote16_t1p0|teacher_ce|1e-6|3|34|0|0|16|1.0|2|1.0"
)

cd lcasr
setting_index=0
for setting in "${SETTINGS[@]}"; do
  setting_index=$((setting_index + 1))
  if [ "${MAX_SETTINGS:-0}" -gt 0 ] && [ "$setting_index" -gt "${MAX_SETTINGS}" ]; then
    echo "MAX_SETTINGS=${MAX_SETTINGS}; stopping after $((setting_index - 1)) settings" | tee -a "${LCASR_LOG_PATH}"
    break
  fi

  IFS='|' read -r name training_mode lr epochs freq_param n_freq n_time vote_samples vote_temp vote_min vote_sim <<< "$setting"
  setting_results="${LCASR_RESULTS_ROOT}/${name}"
  setting_log="${setting_results}/run.log"
  mkdir -p "$setting_results"

  {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${name}"
    echo "training_mode=${training_mode}"
    echo "lr=${lr}"
    echo "epochs=${epochs}"
    echo "spec_augment_freq_mask_param=${freq_param}"
    echo "spec_augment_n_freq_masks=${n_freq}"
    echo "spec_augment_n_time_masks=${n_time}"
    echo "teacher_vote_num_samples=${vote_samples}"
    echo "teacher_vote_temperature=${vote_temp}"
    echo "teacher_vote_min_count=${vote_min}"
    echo "teacher_vote_similarity=${vote_sim}"
    echo "setting_results=${setting_results}"
  } | tee -a "${LCASR_LOG_PATH}" "$setting_log"

  python3.10 results/enc_dec/enc_dec_majority_vote/run_segmented_tedlium_diagnostic.py \
    --output-dir "${setting_results}" \
    --split dev \
    --checkpoint /store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt \
    --seq-len 2048 \
    --overlap 0 \
    --epochs "${epochs}" \
    --training-mode "${training_mode}" \
    --optim-lr "${lr}" \
    --teacher-kl-temperature 1.0 \
    --teacher-vote-num-samples "${vote_samples}" \
    --teacher-vote-temperature "${vote_temp}" \
    --teacher-vote-min-count "${vote_min}" \
    --teacher-vote-similarity "${vote_sim}" \
    --teacher-vote-representative-strategy medoid \
    --enc-dec-beam-width 5 \
    --enc-dec-length-penalty 0.5 \
    --spec-augment-freq-mask-param "${freq_param}" \
    --spec-augment-n-time-masks "${n_time}" \
    --spec-augment-n-freq-masks "${n_freq}" \
    --teacher_filter_max_length \
    --teacher_filter_max_consecutive_token_repeat \
    --teacher_filter_repeated_token_ngrams \
    --teacher_repeated_token_ngram_sizes 2 3 \
    --teacher_filter_repeated_words \
    --teacher_filter_ctc_agreement \
    --max-utterances "${MAX_UTTERANCES:-0}" \
    --max-duration "${MAX_DURATION:-0}" \
    2>&1 | tee -a "${LCASR_LOG_PATH}" "$setting_log"
done
