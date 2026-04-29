#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
CHECKPOINT=${CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt"}
DATASET=${DATASET:-tedlium}
SPLIT=${SPLIT:-dev}
SEQ=${SEQ:-2048}
OVERLAP=${OVERLAP:-0}
DRY_RUN=${DRY_RUN:-0}
RUN_CODEX_ANALYSIS=${RUN_CODEX_ANALYSIS:-0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

RESULTS_DIR="./results/enc_dec_beam_tedlium_dev"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$MPLCONFIGDIR"

CONFIGS=(
  "default|default|__none__"
  "beam5_lp0|--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0"
  "beam5_lp0p2|--enc_dec_beam_width 5 --enc_dec_length_penalty 0.2"
  "beam5_lp0p5|--enc_dec_beam_width 5 --enc_dec_length_penalty 0.5"
  "beam5_lp1p0|--enc_dec_beam_width 5 --enc_dec_length_penalty 1.0"
  "beam5_eos0p5|--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0 --enc_dec_eos_bias 0.5"
  "beam5_eos1p0|--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0 --enc_dec_eos_bias 1.0"
  "beam5_rep0p5_ng3|--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0 --enc_dec_repetition_penalty 0.5 --enc_dec_no_repeat_ngram_size 3"
  "beam5_rep1p0_ng3|--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0 --enc_dec_repetition_penalty 1.0 --enc_dec_no_repeat_ngram_size 3"
  "beam5_ng3|--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0 --enc_dec_no_repeat_ngram_size 3"
  "beam5_eos0p5_rep0p5_ng3|--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0 --enc_dec_eos_bias 0.5 --enc_dec_repetition_penalty 0.5 --enc_dec_no_repeat_ngram_size 3"
  "beam5_max80|--enc_dec_beam_width 5 --enc_dec_length_penalty 0.0 --enc_dec_max_generate 80"
)

for config in "${CONFIGS[@]}"
do
  IFS='|' read -r name decoding_mode arg_string <<< "$config"
  if [ "${arg_string:-}" = "__none__" ]; then
    arg_string=""
  elif [ -z "${arg_string:-}" ]; then
    arg_string="$decoding_mode"
    decoding_mode="beam"
  fi
  read -r -a extra_args <<< "$arg_string"

  run_name="${DATASET}-${SPLIT}-${name}-seq${SEQ}-overlap${OVERLAP}"
  save_path="${RESULTS_DIR}/${run_name}.pkl"
  log_path="${LOG_DIR}/${run_name}.log"

  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${run_name}" | tee "$log_path"
  echo "checkpoint=${CHECKPOINT}" | tee -a "$log_path"
  echo "save_path=${save_path}" | tee -a "$log_path"
  echo "args=${arg_string}" | tee -a "$log_path"

  cmd=(
    "$PYTHON_BIN" enc_dec_inference_test.py
    -c "$CHECKPOINT"
    -dfa
    --dataset "$DATASET"
    --split "$SPLIT"
    --decoding_mode "$decoding_mode"
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

if [ "$DRY_RUN" != "1" ] && [ "$RUN_CODEX_ANALYSIS" = "1" ]; then
  codex exec --cd "$(pwd)" --sandbox workspace-write - <<PROMPT
The TEDLIUM dev encoder-decoder beam-search sweep has finished.

Analyze the outputs in:

${RESULTS_DIR}

Logs are in:

${LOG_DIR}

Create a Markdown report at:

${RESULTS_DIR}/OUTCOME.md

Instructions:
- Do not rerun evaluations.
- Use the saved pickle files and logs to summarize each setting.
- Extract WER and any available insertion/deletion/substitution rates.
- Compare the decoded model outputs across settings, focusing on repetition, early stopping, truncation, and obvious qualitative failures.
- Identify the best setting by WER, then note any setting that is qualitatively safer even if WER is slightly worse.
- Include a compact table with run name, arguments, WER, and notes.
- End with a short recommendation for the next sweep.
- Use python3.10 for any local parsing scripts or one-off commands.
PROMPT
fi
