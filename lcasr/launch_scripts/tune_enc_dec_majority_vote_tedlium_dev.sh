#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
CHECKPOINT=${CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt"}
RESULTS_DIR=${RESULTS_DIR:-"./results/enc_dec/enc_dec_majority_vote"}
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
TEACHER_VOTE_NUM_SAMPLES=${TEACHER_VOTE_NUM_SAMPLES:-8}
TEACHER_VOTE_INCLUDE_DETERMINISTIC=${TEACHER_VOTE_INCLUDE_DETERMINISTIC:-0}
TEACHER_VOTE_REPRESENTATIVE_STRATEGY=${TEACHER_VOTE_REPRESENTATIVE_STRATEGY:-first}
TEACHER_FILTER_LOW_CONFIDENCE=${TEACHER_FILTER_LOW_CONFIDENCE:-0}
TEACHER_MIN_MEAN_MAX_PROB=${TEACHER_MIN_MEAN_MAX_PROB:-0.35}
TEACHER_MAX_MEAN_ENTROPY=${TEACHER_MAX_MEAN_ENTROPY:-2.5}
RUN_BASELINE=${RUN_BASELINE:-1}
DRY_RUN=${DRY_RUN:-0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/exp/exp4/acp21rjf/.scratch/matplotlib}

TRAINING_MODES_STR=${TRAINING_MODES:-"teacher_ce teacher_kl"}
LRS_STR=${LRS:-"1e-7 3e-7"}
KL_TEMPS_STR=${KL_TEMPS:-"1.0"}
VOTE_TEMPS_STR=${VOTE_TEMPS:-"0.7 1.0"}
VOTE_MIN_COUNTS_STR=${VOTE_MIN_COUNTS:-"3"}
VOTE_SIMILARITIES_STR=${VOTE_SIMILARITIES:-"1.0 0.9"}
AUGS_STR=${AUGS:-"freq3_width24_time0 no_aug"}

read -r -a TRAINING_MODES <<< "$TRAINING_MODES_STR"
read -r -a LRS <<< "$LRS_STR"
read -r -a KL_TEMPS <<< "$KL_TEMPS_STR"
read -r -a VOTE_TEMPS <<< "$VOTE_TEMPS_STR"
read -r -a VOTE_MIN_COUNTS <<< "$VOTE_MIN_COUNTS_STR"
read -r -a VOTE_SIMILARITIES <<< "$VOTE_SIMILARITIES_STR"
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
fi

TEACHER_FILTER_ARGS=(
    --teacher_filter_max_length
    --teacher_filter_max_consecutive_token_repeat
    --teacher_filter_repeated_token_ngrams
    --teacher_repeated_token_ngram_sizes 2 3
    --teacher_filter_repeated_words
    --teacher_filter_ctc_agreement
)

VOTE_INCLUDE_ARGS=()
if [ "$TEACHER_VOTE_INCLUDE_DETERMINISTIC" = "1" ]; then
    VOTE_INCLUDE_ARGS=(--teacher_vote_include_deterministic)
fi

CONFIDENCE_FILTER_ARGS=()
CONFIDENCE_TAG=""
if [ "$TEACHER_FILTER_LOW_CONFIDENCE" = "1" ]; then
    CONFIDENCE_FILTER_ARGS=(
        --teacher_filter_low_confidence
        --teacher_min_mean_max_prob "$TEACHER_MIN_MEAN_MAX_PROB"
        --teacher_max_mean_entropy "$TEACHER_MAX_MEAN_ENTROPY"
    )
    CONFIDENCE_TAG="-confp$(tag_value "$TEACHER_MIN_MEAN_MAX_PROB")_e$(tag_value "$TEACHER_MAX_MEAN_ENTROPY")"
fi

run_cmd() {
    local log_path="$1"
    shift
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
    run_cmd "$log_path" "${cmd[@]}"
fi

for training_mode in "${TRAINING_MODES[@]}"
do
    for lr in "${LRS[@]}"
    do
        for kl_temp in "${KL_TEMPS[@]}"
        do
            for vote_temp in "${VOTE_TEMPS[@]}"
            do
                for vote_min_count in "${VOTE_MIN_COUNTS[@]}"
                do
                    for vote_similarity in "${VOTE_SIMILARITIES[@]}"
                    do
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
                                *)
                                    echo "Unknown augmentation setting: $aug" >&2
                                    exit 1
                                    ;;
                            esac

                            lr_tag=$(tag_value "$lr")
                            kl_tag=$(tag_value "$kl_temp")
                            vote_temp_tag=$(tag_value "$vote_temp")
                            vote_sim_tag=$(tag_value "$vote_similarity")
                            vote_tag="voteN${TEACHER_VOTE_NUM_SAMPLES}_t${vote_temp_tag}_min${vote_min_count}_sim${vote_sim_tag}_rep${TEACHER_VOTE_REPRESENTATIVE_STRATEGY}${CONFIDENCE_TAG}"
                            run_name="${DATASET}-${SPLIT}-${training_mode}${DECODING_SUFFIX}-epoch-${EPOCHS}-lr-${lr_tag}-${aug}-${vote_tag}"
                            if [ "$training_mode" = "teacher_kl" ]; then
                                run_name="${run_name}-tau${kl_tag}"
                            fi
                            save_path="${RESULTS_DIR}/${run_name}.pkl"
                            log_path="${LOG_DIR}/${run_name}.log"

                            echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${run_name}" | tee "$log_path"
                            echo "checkpoint=${CHECKPOINT}" | tee -a "$log_path"
                            echo "save_path=${save_path}" | tee -a "$log_path"
                            echo "training_mode=${training_mode}" | tee -a "$log_path"
                            echo "lr=${lr}" | tee -a "$log_path"
                            echo "teacher_kl_temperature=${kl_temp}" | tee -a "$log_path"
                            echo "teacher_vote_num_samples=${TEACHER_VOTE_NUM_SAMPLES}" | tee -a "$log_path"
                            echo "teacher_vote_temperature=${vote_temp}" | tee -a "$log_path"
                            echo "teacher_vote_min_count=${vote_min_count}" | tee -a "$log_path"
                            echo "teacher_vote_similarity=${vote_similarity}" | tee -a "$log_path"
                            echo "teacher_vote_representative_strategy=${TEACHER_VOTE_REPRESENTATIVE_STRATEGY}" | tee -a "$log_path"
                            echo "teacher_filter_low_confidence=${TEACHER_FILTER_LOW_CONFIDENCE}" | tee -a "$log_path"
                            echo "teacher_min_mean_max_prob=${TEACHER_MIN_MEAN_MAX_PROB}" | tee -a "$log_path"
                            echo "teacher_max_mean_entropy=${TEACHER_MAX_MEAN_ENTROPY}" | tee -a "$log_path"
                            echo "augmentation=${aug} kwargs=${aug_kwargs[*]}" | tee -a "$log_path"

                            cmd=(
                                "$PYTHON_BIN" enc_dec_dynamic_eval_test.py
                                "${TEACHER_FILTER_ARGS[@]}"
                                "${CONFIDENCE_FILTER_ARGS[@]}"
                                --training_mode "$training_mode"
                                --teacher_kl_temperature "$kl_temp"
                                --teacher_vote_num_samples "$TEACHER_VOTE_NUM_SAMPLES"
                                --teacher_vote_temperature "$vote_temp"
                                --teacher_vote_min_count "$vote_min_count"
                                --teacher_vote_similarity "$vote_similarity"
                                --teacher_vote_representative_strategy "$TEACHER_VOTE_REPRESENTATIVE_STRATEGY"
                                "${VOTE_INCLUDE_ARGS[@]}"
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
                            run_cmd "$log_path" "${cmd[@]}"
                        done
                    done
                done
            done
        done
    done
done
