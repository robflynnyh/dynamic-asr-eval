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

RESULTS_DIR="./results/enc_dec/checkpoint1/enc_dec_beam_tedlium_dev_refine"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$MPLCONFIGDIR"

CONFIGS=(
  "beam5_lp0p5_ng8|--enc_dec_beam_width 5 --enc_dec_length_penalty 0.5 --enc_dec_no_repeat_ngram_size 8"
  "beam3_lp0p5|--enc_dec_beam_width 3 --enc_dec_length_penalty 0.5"
  "beam10_lp0p5|--enc_dec_beam_width 10 --enc_dec_length_penalty 0.5"
)

for config in "${CONFIGS[@]}"
do
  IFS='|' read -r name arg_string <<< "$config"
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
    --decoding_mode beam
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
The TEDLIUM dev encoder-decoder beam-search refinement sweep has finished.

Analyze the outputs in:

${RESULTS_DIR}

Logs are in:

${LOG_DIR}

Create a Markdown report at:

${RESULTS_DIR}/OUTCOME.md

Instructions:
- Do not rerun evaluations.
- Use the saved pickle files and logs to summarize each setting.
- If needed, copy or adapt results/enc_dec/checkpoint1/enc_dec_beam_tedlium_dev/aggregate.py to aggregate the refinement pickles.
- Extract WER and any available insertion/deletion/substitution rates.
- Compare against results/enc_dec/checkpoint1/enc_dec_beam_tedlium_dev/summary.csv when available.
- Focus on whether no-repeat ngram size 8 acts as a safe repetition guard and whether beam 3 or beam 10 improves over beam 5.
- Include a compact table with run name, arguments, WER, and notes.
- End with a short recommendation.
- Use python3.10 for any local parsing scripts or one-off commands.
PROMPT
fi
