#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-1}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
CHECKPOINT=${CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt"}
SPLIT=${SPLIT:-test}
EPOCHS=${EPOCHS:-1}
REPEATS=${REPEATS:-1}
SEQ=${SEQ:-2048}
OVERLAP=${OVERLAP:-0}
ENC_DEC_BEAM_WIDTH=${ENC_DEC_BEAM_WIDTH:-1}
ENC_DEC_LENGTH_PENALTY=${ENC_DEC_LENGTH_PENALTY:-0.0}
ENC_DEC_EOS_BIAS=${ENC_DEC_EOS_BIAS:-0.0}
ENC_DEC_REPETITION_PENALTY=${ENC_DEC_REPETITION_PENALTY:-0.0}
ENC_DEC_NO_REPEAT_NGRAM_SIZE=${ENC_DEC_NO_REPEAT_NGRAM_SIZE:-0}
ENC_DEC_MAX_GENERATE=${ENC_DEC_MAX_GENERATE:--1}
TRAINING_MODE=${TRAINING_MODE:-teacher_ce}
TEACHER_KL_TEMPERATURE=${TEACHER_KL_TEMPERATURE:-1.0}
TEACHER_FILTER_CTC_AGREEMENT=${TEACHER_FILTER_CTC_AGREEMENT:-1}
TEACHER_REPEATED_TOKEN_NGRAM_SIZES=${TEACHER_REPEATED_TOKEN_NGRAM_SIZES:-"2 3"}
TEACHER_FILTER_LOW_CONFIDENCE=${TEACHER_FILTER_LOW_CONFIDENCE:-0}
TEACHER_MIN_MEAN_MAX_PROB=${TEACHER_MIN_MEAN_MAX_PROB:-0.35}
TEACHER_MAX_MEAN_ENTROPY=${TEACHER_MAX_MEAN_ENTROPY:-2.5}
TEACHER_EPOCH_RELABEL=${TEACHER_EPOCH_RELABEL:-0}
DRY_RUN=${DRY_RUN:-0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

DATASETS_STR=${DATASETS:-"tedlium earnings22"}
LRS_STR=${LRS:-"3e-7 1e-7 1e-6"}
AUGS_STR=${AUGS:-"freq6_width34_time0 freq3_width24_time0 no_aug"}

read -r -a DATASETS <<< "$DATASETS_STR"
read -r -a LRS <<< "$LRS_STR"
read -r -a AUGS <<< "$AUGS_STR"

RESULTS_DIR=${RESULTS_DIR:-"./results/enc_dec_dynamic_eval"}
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$MPLCONFIGDIR"

lr_tag() {
    echo "$1" | sed 's/-/m/g; s/+//g; s/\./p/g'
}

read -r -a TEACHER_REPEATED_TOKEN_NGRAM_SIZES_ARR <<< "$TEACHER_REPEATED_TOKEN_NGRAM_SIZES"

TEACHER_FILTER_ARGS=(
    --teacher_filter_max_length
    --teacher_filter_max_consecutive_token_repeat
    --teacher_filter_repeated_token_ngrams
    --teacher_repeated_token_ngram_sizes "${TEACHER_REPEATED_TOKEN_NGRAM_SIZES_ARR[@]}"
    --teacher_filter_repeated_words
)
if [ "$TEACHER_FILTER_CTC_AGREEMENT" = "1" ]; then
    TEACHER_FILTER_ARGS+=(--teacher_filter_ctc_agreement)
fi
if [ "$TEACHER_FILTER_LOW_CONFIDENCE" = "1" ]; then
    TEACHER_FILTER_ARGS+=(
        --teacher_filter_low_confidence
        --teacher_min_mean_max_prob "$TEACHER_MIN_MEAN_MAX_PROB"
        --teacher_max_mean_entropy "$TEACHER_MAX_MEAN_ENTROPY"
    )
fi
TEACHER_EPOCH_RELABEL_ARGS=()
RUN_TRAINING_MODE="$TRAINING_MODE"
if [ "$TEACHER_EPOCH_RELABEL" = "1" ]; then
    TEACHER_EPOCH_RELABEL_ARGS+=(--teacher_epoch_relabel)
    RUN_TRAINING_MODE="${TRAINING_MODE}_epoch_relabel"
fi

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
    DECODING_SUFFIX="-beam${ENC_DEC_BEAM_WIDTH}_lp$(lr_tag "$ENC_DEC_LENGTH_PENALTY")"
    if [ "$ENC_DEC_NO_REPEAT_NGRAM_SIZE" != "0" ]; then
        DECODING_SUFFIX="${DECODING_SUFFIX}_ng${ENC_DEC_NO_REPEAT_NGRAM_SIZE}"
    fi
    if [ "$ENC_DEC_EOS_BIAS" != "0.0" ]; then
        DECODING_SUFFIX="${DECODING_SUFFIX}_eos$(lr_tag "$ENC_DEC_EOS_BIAS")"
    fi
    if [ "$ENC_DEC_REPETITION_PENALTY" != "0.0" ]; then
        DECODING_SUFFIX="${DECODING_SUFFIX}_rep$(lr_tag "$ENC_DEC_REPETITION_PENALTY")"
    fi
    if [ "$ENC_DEC_MAX_GENERATE" != "-1" ]; then
        DECODING_SUFFIX="${DECODING_SUFFIX}_max${ENC_DEC_MAX_GENERATE}"
    fi
fi

for dataset in "${DATASETS[@]}"
do
    for lr in "${LRS[@]}"
    do
        for aug in "${AUGS[@]}"
        do
            case "$aug" in
                freq6_width34_time0)
                    aug_kwargs=(spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6)
                    ;;
                freq3_width24_time0)
                    aug_kwargs=(spec_augment_freq_mask_param=24 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=3)
                    ;;
                no_aug)
                    aug_kwargs=(spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=0)
                    ;;
                *)
                    echo "Unknown augmentation setting: $aug" >&2
                    exit 1
                    ;;
            esac

            lr_name=$(lr_tag "$lr")
            run_name="${dataset}-${SPLIT}-${RUN_TRAINING_MODE}${DECODING_SUFFIX}-epoch-${EPOCHS}-lr-${lr_name}-${aug}"
            save_path="${RESULTS_DIR}/${run_name}.pkl"
            log_path="${LOG_DIR}/${run_name}.log"

            echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${run_name}" | tee "$log_path"
            echo "checkpoint=${CHECKPOINT}" | tee -a "$log_path"
            echo "save_path=${save_path}" | tee -a "$log_path"
            echo "augmentation=${aug} kwargs=${aug_kwargs[*]}" | tee -a "$log_path"
            echo "decoding_args=${DECODING_ARGS[*]}" | tee -a "$log_path"
            echo "teacher_kl_temperature=${TEACHER_KL_TEMPERATURE}" | tee -a "$log_path"
            echo "teacher_filter_ctc_agreement=${TEACHER_FILTER_CTC_AGREEMENT}" | tee -a "$log_path"
            echo "teacher_repeated_token_ngram_sizes=${TEACHER_REPEATED_TOKEN_NGRAM_SIZES}" | tee -a "$log_path"
            echo "teacher_filter_low_confidence=${TEACHER_FILTER_LOW_CONFIDENCE}" | tee -a "$log_path"
            echo "teacher_min_mean_max_prob=${TEACHER_MIN_MEAN_MAX_PROB}" | tee -a "$log_path"
            echo "teacher_max_mean_entropy=${TEACHER_MAX_MEAN_ENTROPY}" | tee -a "$log_path"
            echo "teacher_epoch_relabel=${TEACHER_EPOCH_RELABEL}" | tee -a "$log_path"

            cmd=(
                "$PYTHON_BIN" enc_dec_dynamic_eval_test.py
                "${TEACHER_FILTER_ARGS[@]}" \
                --training_mode "$TRAINING_MODE" \
                --teacher_kl_temperature "$TEACHER_KL_TEMPERATURE" \
                "${TEACHER_EPOCH_RELABEL_ARGS[@]}" \
                -c "$CHECKPOINT" \
                -dfa \
                -epochs "$EPOCHS" \
                -r "$REPEATS" \
                -seq "$SEQ" \
                -o "$OVERLAP" \
                --split "$SPLIT" \
                --dataset "$dataset" \
                -s "$save_path" \
                -log "$log_path" \
                "${DECODING_ARGS[@]}" \
                -kwargs optim_lr="$lr" "${aug_kwargs[@]}"
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
