#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU=${GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
CHECKPOINT=${CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt"}
REPEATS=${REPEATS:-1}
SEQ=${SEQ:-2048}
OVERLAP=${OVERLAP:-0}
ENC_DEC_BEAM_WIDTH=${ENC_DEC_BEAM_WIDTH:-5}
ENC_DEC_LENGTH_PENALTY=${ENC_DEC_LENGTH_PENALTY:-0.5}
ENC_DEC_EOS_BIAS=${ENC_DEC_EOS_BIAS:-0.0}
ENC_DEC_REPETITION_PENALTY=${ENC_DEC_REPETITION_PENALTY:-0.0}
ENC_DEC_NO_REPEAT_NGRAM_SIZE=${ENC_DEC_NO_REPEAT_NGRAM_SIZE:-0}
ENC_DEC_MAX_GENERATE=${ENC_DEC_MAX_GENERATE:--1}
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

RESULTS_DIR=${RESULTS_DIR:-"./results/enc_dec/enc_dec_v2/rob97_unadapted_beam"}
PKL_DIR="${RESULTS_DIR}/pkl"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "$PKL_DIR" "$LOG_DIR" "$MPLCONFIGDIR"

read -r -d '' DEFAULT_RUN_MATRIX <<'EOF' || true
chime6 test
rev16 test
EOF
RUN_MATRIX=${RUN_MATRIX:-$DEFAULT_RUN_MATRIX}

tag_value() {
    echo "$1" | sed 's/-/m/g; s/+//g; s/\./p/g'
}

decoding_args=(
    --enc_dec_beam_width "$ENC_DEC_BEAM_WIDTH"
    --enc_dec_length_penalty "$ENC_DEC_LENGTH_PENALTY"
    --enc_dec_eos_bias "$ENC_DEC_EOS_BIAS"
    --enc_dec_repetition_penalty "$ENC_DEC_REPETITION_PENALTY"
    --enc_dec_no_repeat_ngram_size "$ENC_DEC_NO_REPEAT_NGRAM_SIZE"
    --enc_dec_max_generate "$ENC_DEC_MAX_GENERATE"
)
decode_suffix="beam${ENC_DEC_BEAM_WIDTH}_lp$(tag_value "$ENC_DEC_LENGTH_PENALTY")"
if [ "$ENC_DEC_NO_REPEAT_NGRAM_SIZE" != "0" ]; then
    decode_suffix="${decode_suffix}_ng${ENC_DEC_NO_REPEAT_NGRAM_SIZE}"
fi
if [ "$ENC_DEC_EOS_BIAS" != "0.0" ]; then
    decode_suffix="${decode_suffix}_eos$(tag_value "$ENC_DEC_EOS_BIAS")"
fi
if [ "$ENC_DEC_REPETITION_PENALTY" != "0.0" ]; then
    decode_suffix="${decode_suffix}_rep$(tag_value "$ENC_DEC_REPETITION_PENALTY")"
fi
if [ "$ENC_DEC_MAX_GENERATE" != "-1" ]; then
    decode_suffix="${decode_suffix}_max${ENC_DEC_MAX_GENERATE}"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

while read -r dataset split
do
    if [ -z "${dataset:-}" ] || [[ "$dataset" == \#* ]]; then
        continue
    fi

    run_name="${dataset}-${split}-enc_dec_v2-no_adapt-${decode_suffix}-epoch-0-lr-none-no_aug"
    save_path="${PKL_DIR}/${run_name}.pkl"
    log_path="${LOG_DIR}/${run_name}.log"

    existing_repeats=1
    for repeat_id in $(seq 1 "$REPEATS")
    do
        repeated_save_path="${save_path%.pkl}_${repeat_id}.pkl"
        if [ ! -s "$repeated_save_path" ]; then
            existing_repeats=0
        fi
    done
    if [ "$SKIP_EXISTING" = "1" ] && { [ -s "$save_path" ] || [ "$existing_repeats" = "1" ]; }; then
        {
            echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] skipping existing ${run_name}"
            echo "save_path=${save_path}"
        } | tee -a "$log_path"
        continue
    fi

    {
        echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${run_name}"
        echo "gpu=${GPU}"
        echo "checkpoint=${CHECKPOINT}"
        echo "save_path=${save_path}"
        echo "dataset=${dataset}"
        echo "split=${split}"
        echo "training_mode=no_adapt"
        echo "epochs=0"
        echo "decode_args=${decoding_args[*]}"
    } | tee "$log_path"

    cmd=(
        "$PYTHON_BIN" enc_dec_dynamic_eval_test.py
        --training_mode teacher_ce
        -c "$CHECKPOINT"
        -dfa
        -epochs 0
        -r "$REPEATS"
        -seq "$SEQ"
        -o "$OVERLAP"
        --split "$split"
        --dataset "$dataset"
        -s "$save_path"
        -log "$log_path"
        "${decoding_args[@]}"
        -kwargs optim_lr=0.0 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=0
    )

    if [ "$DRY_RUN" = "1" ]; then
        printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU" | tee -a "$log_path"
        printf ' %q' "${cmd[@]}" | tee -a "$log_path"
        printf '\n' | tee -a "$log_path"
    else
        CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" 2>&1 | tee -a "$log_path"
    fi
done <<< "$RUN_MATRIX"

if [ "$DRY_RUN" != "1" ]; then
    "$PYTHON_BIN" results/enc_dec/enc_dec_v2/rob97_unadapted_beam/aggregate.py \
        --csv "${RESULTS_DIR}/summary.csv" \
        --outcome "${RESULTS_DIR}/OUTCOME.md"
fi
