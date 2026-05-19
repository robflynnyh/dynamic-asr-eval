#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-1}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
CHECKPOINT=${CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt"}
SPLIT=${SPLIT:-test}
REPEATS=${REPEATS:-1}
SEQ=${SEQ:-2048}
OVERLAP=${OVERLAP:-0}
ENC_DEC_BEAM_WIDTH=${ENC_DEC_BEAM_WIDTH:-1}
ENC_DEC_LENGTH_PENALTY=${ENC_DEC_LENGTH_PENALTY:-0.0}
ENC_DEC_EOS_BIAS=${ENC_DEC_EOS_BIAS:-0.0}
ENC_DEC_REPETITION_PENALTY=${ENC_DEC_REPETITION_PENALTY:-0.0}
ENC_DEC_NO_REPEAT_NGRAM_SIZE=${ENC_DEC_NO_REPEAT_NGRAM_SIZE:-0}
ENC_DEC_MAX_GENERATE=${ENC_DEC_MAX_GENERATE:--1}
DRY_RUN=${DRY_RUN:-0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

DATASETS_STR=${DATASETS:-"tedlium earnings22"}
read -r -a DATASETS <<< "$DATASETS_STR"

RESULTS_DIR="./results/enc_dec/checkpoint1/enc_dec_dynamic_eval"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$MPLCONFIGDIR"

tag_value() {
    echo "$1" | sed 's/-/m/g; s/+//g; s/\./p/g'
}

DECODING_ARGS=()
DECODING_SUFFIX=""
if [ "$ENC_DEC_BEAM_WIDTH" != "1" ]; then
    DECODING_ARGS=(
        --enc_dec_beam_width "$ENC_DEC_BEAM_WIDTH"
        --enc_dec_length_penalty "$ENC_DEC_LENGTH_PENALTY"
        --enc_dec_eos_bias "$ENC_DEC_EOS_BIAS"
        --enc_dec_repetition_penalty "$ENC_DEC_REPETITION_PENALTY"
        --enc_dec_no_repeat_ngram_size "$ENC_DEC_NO_REPEAT_NGRAM_SIZE"
        --enc_dec_max_generate "$ENC_DEC_MAX_GENERATE"
    )
    DECODING_SUFFIX="-beam${ENC_DEC_BEAM_WIDTH}_lp$(tag_value "$ENC_DEC_LENGTH_PENALTY")"
    if [ "$ENC_DEC_NO_REPEAT_NGRAM_SIZE" != "0" ]; then
        DECODING_SUFFIX="${DECODING_SUFFIX}_ng${ENC_DEC_NO_REPEAT_NGRAM_SIZE}"
    fi
    if [ "$ENC_DEC_EOS_BIAS" != "0.0" ]; then
        DECODING_SUFFIX="${DECODING_SUFFIX}_eos$(tag_value "$ENC_DEC_EOS_BIAS")"
    fi
    if [ "$ENC_DEC_REPETITION_PENALTY" != "0.0" ]; then
        DECODING_SUFFIX="${DECODING_SUFFIX}_rep$(tag_value "$ENC_DEC_REPETITION_PENALTY")"
    fi
    if [ "$ENC_DEC_MAX_GENERATE" != "-1" ]; then
        DECODING_SUFFIX="${DECODING_SUFFIX}_max${ENC_DEC_MAX_GENERATE}"
    fi
fi

for dataset in "${DATASETS[@]}"
do
    run_name="${dataset}-${SPLIT}-no_adapt${DECODING_SUFFIX}-epoch-0-lr-none-no_aug"
    save_path="${RESULTS_DIR}/${run_name}.pkl"
    log_path="${LOG_DIR}/${run_name}.log"

    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${run_name}" | tee "$log_path"
    echo "checkpoint=${CHECKPOINT}" | tee -a "$log_path"
    echo "save_path=${save_path}" | tee -a "$log_path"
    echo "decoding_args=${DECODING_ARGS[*]}" | tee -a "$log_path"

    cmd=(
        "$PYTHON_BIN" enc_dec_dynamic_eval_test.py
        --training_mode teacher_ce
        -c "$CHECKPOINT"
        -dfa
        -epochs 0
        -r "$REPEATS"
        -seq "$SEQ"
        -o "$OVERLAP"
        --split "$SPLIT"
        --dataset "$dataset"
        -s "$save_path"
        -log "$log_path"
        "${DECODING_ARGS[@]}"
        -kwargs optim_lr=0.0 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=0
    )

    if [ "$DRY_RUN" = "1" ]; then
        printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU" | tee -a "$log_path"
        printf ' %q' "${cmd[@]}" | tee -a "$log_path"
        printf '\n' | tee -a "$log_path"
    else
        CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" 2>&1 | tee -a "$log_path"
    fi
done
