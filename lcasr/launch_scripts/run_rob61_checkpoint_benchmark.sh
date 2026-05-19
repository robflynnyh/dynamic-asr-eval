#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU=${GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
SPLIT=${SPLIT:-test}
SEQ=${SEQ:-2048}
OVERLAP=${OVERLAP:-0}
DRY_RUN=${DRY_RUN:-0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

OLD_CHECKPOINT=${OLD_CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt"}
NEW_CHECKPOINT_ROOT=${NEW_CHECKPOINT_ROOT:-"/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5"}
NEW_STEPS_STR=${NEW_STEPS:-"2000 10000 20000 30000"}
DATASETS_STR=${DATASETS:-"tedlium earnings22"}
DECODE_CONFIGS_STR=${DECODE_CONFIGS:-"greedy beam5_lp0p5"}

RESULTS_ROOT=${RESULTS_ROOT:-"./results/enc_dec/checkpoint2/rob61_checkpoint_benchmark"}
PKL_ROOT="${RESULTS_ROOT}/pkl"
LOG_ROOT="${RESULTS_ROOT}/logs"
mkdir -p "$PKL_ROOT" "$LOG_ROOT" "$MPLCONFIGDIR"

read -r -a NEW_STEPS <<< "$NEW_STEPS_STR"
read -r -a DATASETS <<< "$DATASETS_STR"
read -r -a DECODE_CONFIGS <<< "$DECODE_CONFIGS_STR"

CHECKPOINT_LABELS=("old_seed")
CHECKPOINT_PATHS=("$OLD_CHECKPOINT")
for step in "${NEW_STEPS[@]}"
do
  CHECKPOINT_LABELS+=("rl_step_${step}")
  CHECKPOINT_PATHS+=("${NEW_CHECKPOINT_ROOT}/step_${step}.pt")
done

decode_args() {
  local decode="$1"
  case "$decode" in
    greedy)
      printf '%s\n' "--decoding_mode default"
      ;;
    beam5_lp0p5)
      printf '%s\n' "--decoding_mode beam --enc_dec_beam_width 5 --enc_dec_length_penalty 0.5"
      ;;
    *)
      echo "Unknown decode config: ${decode}" >&2
      return 1
      ;;
  esac
}

for checkpoint_index in "${!CHECKPOINT_LABELS[@]}"
do
  label="${CHECKPOINT_LABELS[$checkpoint_index]}"
  checkpoint="${CHECKPOINT_PATHS[$checkpoint_index]}"
  if [ ! -f "$checkpoint" ]; then
    echo "Missing checkpoint for ${label}: ${checkpoint}" >&2
    exit 1
  fi

  for dataset in "${DATASETS[@]}"
  do
    for decode in "${DECODE_CONFIGS[@]}"
    do
      read -r -a extra_args <<< "$(decode_args "$decode")"
      run_name="${dataset}-${SPLIT}-${label}-${decode}-seq${SEQ}-overlap${OVERLAP}"
      save_path="${PKL_ROOT}/${run_name}.pkl"
      log_path="${LOG_ROOT}/${run_name}.log"

      {
        echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${run_name}"
        echo "gpu=${GPU}"
        echo "checkpoint=${checkpoint}"
        echo "save_path=${save_path}"
        echo "decode=${decode}"
        echo "decode_args=${extra_args[*]}"
      } | tee "$log_path"

      cmd=(
        "$PYTHON_BIN" enc_dec_inference_test.py
        -c "$checkpoint"
        -dfa
        --dataset "$dataset"
        --split "$SPLIT"
        -seq "$SEQ"
        -o "$OVERLAP"
        -s "$save_path"
        -log "$log_path"
        "${extra_args[@]}"
      )

      if [ "$DRY_RUN" = "1" ]; then
        printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU" | tee -a "$log_path"
        printf ' %q' "${cmd[@]}" | tee -a "$log_path"
        printf '\n' | tee -a "$log_path"
      else
        CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" 2>&1 | tee -a "$log_path"
      fi
    done
  done
done

if [ "$DRY_RUN" != "1" ]; then
  "$PYTHON_BIN" results/enc_dec/checkpoint2/rob61_checkpoint_benchmark/aggregate.py \
    --csv results/enc_dec/checkpoint2/rob61_checkpoint_benchmark/summary.csv \
    --outcome results/enc_dec/checkpoint2/rob61_checkpoint_benchmark/OUTCOME.md
fi
