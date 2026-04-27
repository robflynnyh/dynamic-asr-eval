#!/usr/bin/env bash
set -euo pipefail

EPOCHS=(1 3)
REPEATS=1
DEVICE="2"
SEQ=2048
OLAP=0
LR="9e-6"
CHECKPOINT="/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt"
RESULTS_DIR="./results/enc_dec_dynamic_eval"
LOG_DIR="${RESULTS_DIR}/logs"
DATASET="earnings22"
SPLIT="test"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

for epoch in "${EPOCHS[@]}"
do
    for variant in baseline repetition_guard quality_guard quality_plus_decode_agreement
    do
        EXTRA_ARGS=()

        case "$variant" in
            baseline)
                ;;
            repetition_guard)
                EXTRA_ARGS=(
                    --teacher_filter_max_length
                    --teacher_filter_max_consecutive_token_repeat
                    --teacher_filter_repeated_token_ngrams
                    --teacher_filter_repeated_words
                )
                ;;
            quality_guard)
                EXTRA_ARGS=(
                    --teacher_filter_max_length
                    --teacher_filter_max_consecutive_token_repeat
                    --teacher_filter_repeated_token_ngrams
                    --teacher_filter_repeated_words
                    --teacher_filter_low_confidence
                    --teacher_filter_ctc_agreement
                )
                ;;
            quality_plus_decode_agreement)
                EXTRA_ARGS=(
                    --teacher_filter_max_length
                    --teacher_filter_max_consecutive_token_repeat
                    --teacher_filter_repeated_token_ngrams
                    --teacher_filter_repeated_words
                    --teacher_filter_low_confidence
                    --teacher_filter_ctc_agreement
                    --teacher_filter_decode_agreement
                )
                ;;
            *)
                echo "Unknown variant: $variant" >&2
                exit 1
                ;;
        esac

        echo "$DATASET $SPLIT epoch=$epoch variant=$variant"
        CUDA_VISIBLE_DEVICES=$DEVICE python3.10 enc_dec_dynamic_eval_test.py \
            -c "$CHECKPOINT" \
            -dfa \
            -epochs "$epoch" \
            -r "$REPEATS" \
            -seq "$SEQ" \
            -o "$OLAP" \
            -split "$SPLIT" \
            --dataset "$DATASET" \
            -kwargs optim_lr=$LR \
            -s "${RESULTS_DIR}/${DATASET}-${SPLIT}-epoch-${epoch}-lr-9em6-${variant}.pkl" \
            -log "${LOG_DIR}/${DATASET}-${SPLIT}-epoch-${epoch}-lr-9em6-${variant}.log" \
            "${EXTRA_ARGS[@]}"
        echo "done"
    done
done
