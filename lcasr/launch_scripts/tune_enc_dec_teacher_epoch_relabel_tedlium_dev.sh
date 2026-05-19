#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
CHECKPOINT=${CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt"}
RESULTS_DIR=${RESULTS_DIR:-"./results/enc_dec/checkpoint1/enc_dec_teacher_epoch_relabel"}
LOG_DIR="${RESULTS_DIR}/logs"
DATASET=${DATASET:-tedlium}
SPLIT=${SPLIT:-dev}
EPOCHS=${EPOCHS:-1}
REPEATS=${REPEATS:-1}
SEQ=${SEQ:-2048}
OVERLAP=${OVERLAP:-0}
ENC_DEC_BEAM_WIDTH=${ENC_DEC_BEAM_WIDTH:-5}
ENC_DEC_LENGTH_PENALTY=${ENC_DEC_LENGTH_PENALTY:-0.5}
ENC_DEC_EOS_BIAS=${ENC_DEC_EOS_BIAS:-0.0}
ENC_DEC_REPETITION_PENALTY=${ENC_DEC_REPETITION_PENALTY:-0.0}
ENC_DEC_NO_REPEAT_NGRAM_SIZE=${ENC_DEC_NO_REPEAT_NGRAM_SIZE:-0}
ENC_DEC_MAX_GENERATE=${ENC_DEC_MAX_GENERATE:--1}
RUN_BASELINE=${RUN_BASELINE:-1}
DRY_RUN=${DRY_RUN:-0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

LRS_STR=${LRS:-"3e-7 1e-7 1e-8"}
KL_TEMPS_STR=${KL_TEMPS:-"1.0 0.7 0.5"}
FILTERS_STR=${FILTERS:-"relaxed ctc strict_ctc"}
AUGS_STR=${AUGS:-"no_aug freq2_width16_time0 freq3_width24_time0 freq6_width34_time0 freq8_width48_time0 freq2_width16_time1"}

read -r -a LRS <<< "$LRS_STR"
read -r -a KL_TEMPS <<< "$KL_TEMPS_STR"
read -r -a FILTERS <<< "$FILTERS_STR"
read -r -a AUGS <<< "$AUGS_STR"

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

run_cmd() {
    local run_name="$1"
    local log_path="$2"
    shift 2

    if [ "$DRY_RUN" = "1" ]; then
        printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU" | tee -a "$log_path"
        printf ' %q' "$@" | tee -a "$log_path"
        printf '\n' | tee -a "$log_path"
    else
        CUDA_VISIBLE_DEVICES="$GPU" "$@" 2>&1 | tee -a "$log_path"
    fi
}

if [ "$RUN_BASELINE" = "1" ]; then
    run_name="${DATASET}-${SPLIT}-no_adapt${DECODING_SUFFIX}-epoch-0-lr-none-no_aug"
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
        --dataset "$DATASET"
        -s "$save_path"
        -log "$log_path"
        "${DECODING_ARGS[@]}"
        -kwargs optim_lr=0.0 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=0
    )
    run_cmd "$run_name" "$log_path" "${cmd[@]}"
fi

for lr in "${LRS[@]}"
do
    for kl_temp in "${KL_TEMPS[@]}"
    do
        for filter in "${FILTERS[@]}"
        do
            teacher_filter_args=(
                --teacher_filter_max_length
                --teacher_filter_max_consecutive_token_repeat
                --teacher_filter_repeated_token_ngrams
                --teacher_filter_repeated_words
            )
            case "$filter" in
                relaxed)
                    teacher_filter_args+=(--teacher_repeated_token_ngram_sizes 6)
                    ;;
                ctc)
                    teacher_filter_args+=(--teacher_repeated_token_ngram_sizes 6 --teacher_filter_ctc_agreement)
                    ;;
                strict_ctc)
                    teacher_filter_args+=(--teacher_repeated_token_ngram_sizes 2 3 --teacher_filter_ctc_agreement)
                    ;;
                lowconf_ctc)
                    teacher_filter_args+=(
                        --teacher_repeated_token_ngram_sizes 6
                        --teacher_filter_ctc_agreement
                        --teacher_filter_low_confidence
                        --teacher_min_mean_max_prob 0.35
                        --teacher_max_mean_entropy 2.5
                    )
                    ;;
                *)
                    echo "Unknown filter setting: $filter" >&2
                    exit 1
                    ;;
            esac

            for aug in "${AUGS[@]}"
            do
                case "$aug" in
                    no_aug)
                        aug_kwargs=(spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=0)
                        ;;
                    freq2_width16_time0)
                        aug_kwargs=(spec_augment_freq_mask_param=16 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=2)
                        ;;
                    freq3_width24_time0)
                        aug_kwargs=(spec_augment_freq_mask_param=24 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=3)
                        ;;
                    freq6_width34_time0)
                        aug_kwargs=(spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6)
                        ;;
                    freq8_width48_time0)
                        aug_kwargs=(spec_augment_freq_mask_param=48 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=8)
                        ;;
                    freq2_width16_time1)
                        aug_kwargs=(spec_augment_freq_mask_param=16 spec_augment_n_time_masks=1 spec_augment_n_freq_masks=2)
                        ;;
                    freq3_width24_time1)
                        aug_kwargs=(spec_augment_freq_mask_param=24 spec_augment_n_time_masks=1 spec_augment_n_freq_masks=3)
                        ;;
                    freq6_width34_time1)
                        aug_kwargs=(spec_augment_freq_mask_param=34 spec_augment_n_time_masks=1 spec_augment_n_freq_masks=6)
                        ;;
                    time1_only)
                        aug_kwargs=(spec_augment_freq_mask_param=34 spec_augment_n_time_masks=1 spec_augment_n_freq_masks=0)
                        ;;
                    *)
                        echo "Unknown augmentation setting: $aug" >&2
                        exit 1
                        ;;
                esac

                lr_tag=$(tag_value "$lr")
                kl_tag=$(tag_value "$kl_temp")
                run_name="${DATASET}-${SPLIT}-teacher_kl_epoch_relabel${DECODING_SUFFIX}-epoch-${EPOCHS}-lr-${lr_tag}-${aug}-tau${kl_tag}-filter_${filter}"
                save_path="${RESULTS_DIR}/${run_name}.pkl"
                log_path="${LOG_DIR}/${run_name}.log"

                echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${run_name}" | tee "$log_path"
                echo "checkpoint=${CHECKPOINT}" | tee -a "$log_path"
                echo "save_path=${save_path}" | tee -a "$log_path"
                echo "lr=${lr}" | tee -a "$log_path"
                echo "teacher_kl_temperature=${kl_temp}" | tee -a "$log_path"
                echo "filter=${filter}" | tee -a "$log_path"
                echo "augmentation=${aug} kwargs=${aug_kwargs[*]}" | tee -a "$log_path"
                echo "decoding_args=${DECODING_ARGS[*]}" | tee -a "$log_path"

                cmd=(
                    "$PYTHON_BIN" enc_dec_dynamic_eval_test.py
                    "${teacher_filter_args[@]}"
                    --training_mode teacher_kl
                    --teacher_epoch_relabel
                    --teacher_kl_temperature "$kl_temp"
                    -c "$CHECKPOINT"
                    -dfa
                    -epochs "$EPOCHS"
                    -r "$REPEATS"
                    -seq "$SEQ"
                    -o "$OVERLAP"
                    --split "$SPLIT"
                    --dataset "$DATASET"
                    -s "$save_path"
                    -log "$log_path"
                    "${DECODING_ARGS[@]}"
                    -kwargs optim_lr="$lr" "${aug_kwargs[@]}"
                )
                run_cmd "$run_name" "$log_path" "${cmd[@]}"
            done
        done
    done
done
