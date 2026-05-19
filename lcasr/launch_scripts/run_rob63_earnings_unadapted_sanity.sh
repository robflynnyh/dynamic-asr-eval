#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU=${GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
SEQ=${SEQ:-2048}
OVERLAP=${OVERLAP:-0}
DRY_RUN=${DRY_RUN:-0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

OLD_CHECKPOINT=${OLD_CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt"}
OUTCOME_CHECKPOINT=${OUTCOME_CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt"}
RESULTS_ROOT=${RESULTS_ROOT:-"./results/enc_dec/enc_dec_v2/rob63_earnings_unadapted_sanity"}
PKL_ROOT="${RESULTS_ROOT}/pkl"
LOG_ROOT="${RESULTS_ROOT}/logs"
mkdir -p "$PKL_ROOT" "$LOG_ROOT" "$MPLCONFIGDIR"

CHECKPOINT_LABELS=("old_seed" "enc_dec_outcome_seed")
CHECKPOINT_PATHS=("$OLD_CHECKPOINT" "$OUTCOME_CHECKPOINT")

for checkpoint_index in "${!CHECKPOINT_LABELS[@]}"
do
  label="${CHECKPOINT_LABELS[$checkpoint_index]}"
  checkpoint="${CHECKPOINT_PATHS[$checkpoint_index]}"
  if [ ! -f "$checkpoint" ]; then
    echo "Missing checkpoint for ${label}: ${checkpoint}" >&2
    exit 1
  fi

  run_name="earnings22-test-${label}-beam5_lp0p5-seq${SEQ}-overlap${OVERLAP}"
  save_path="${PKL_ROOT}/${run_name}_1.pkl"
  log_path="${LOG_ROOT}/${run_name}.log"

  {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${run_name}"
    echo "gpu=${GPU}"
    echo "checkpoint=${checkpoint}"
    echo "save_path=${save_path}"
    echo "decode=beam5_lp0p5"
  } | tee "$log_path"

  cmd=(
    "$PYTHON_BIN" enc_dec_inference_test.py
    -c "$checkpoint"
    -dfa
    --dataset earnings22
    --split test
    -seq "$SEQ"
    -o "$OVERLAP"
    -s "$save_path"
    -log "$log_path"
    --decoding_mode beam
    --enc_dec_beam_width 5
    --enc_dec_length_penalty 0.5
  )

  if [ "$DRY_RUN" = "1" ]; then
    printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU" | tee -a "$log_path"
    printf ' %q' "${cmd[@]}" | tee -a "$log_path"
    printf '\n' | tee -a "$log_path"
  else
    CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" 2>&1 | tee -a "$log_path"
  fi
done

if [ "$DRY_RUN" != "1" ]; then
  "$PYTHON_BIN" results/enc_dec/enc_dec_v2/rob63_earnings_unadapted_sanity/aggregate.py \
    --directory results/enc_dec/enc_dec_v2/rob63_earnings_unadapted_sanity/pkl \
    --csv results/enc_dec/enc_dec_v2/rob63_earnings_unadapted_sanity/summary.csv \
    --outcome results/enc_dec/enc_dec_v2/rob63_earnings_unadapted_sanity/OUTCOME.md
fi
