#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU=${GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
SPLIT=${SPLIT:-test}
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
TEACHER_KL_TEMPERATURE=${TEACHER_KL_TEMPERATURE:-1.0}
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

OLD_CHECKPOINT=${OLD_CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt"}
RL_CHECKPOINT=${RL_CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt"}
CHECKPOINTS_STR=${CHECKPOINTS:-"old_seed rl_step_30000"}
DATASETS_STR=${DATASETS:-"tedlium earnings22"}
TRAINING_MODES_STR=${TRAINING_MODES:-"teacher_ce teacher_kl"}
LRS_STR=${LRS:-"1e-7 3e-7"}
AUGS_STR=${AUGS:-"freq6_width34_time0 freq3_width24_time0"}
FILTERS_STR=${FILTERS:-"no_filter"}

RESULTS_ROOT=${RESULTS_ROOT:-"./results/enc_dec/rl_step_30000/rob63_rl_self_training_compare"}
PKL_ROOT="${RESULTS_ROOT}/pkl"
LOG_ROOT="${RESULTS_ROOT}/logs"
mkdir -p "$PKL_ROOT" "$LOG_ROOT" "$MPLCONFIGDIR"

read -r -a DATASETS <<< "$DATASETS_STR"
read -r -a TRAINING_MODES <<< "$TRAINING_MODES_STR"
read -r -a LRS <<< "$LRS_STR"
read -r -a AUGS <<< "$AUGS_STR"
read -r -a FILTERS <<< "$FILTERS_STR"
read -r -a REQUESTED_CHECKPOINTS <<< "$CHECKPOINTS_STR"

CHECKPOINT_LABELS=()
CHECKPOINT_PATHS=()
for requested_checkpoint in "${REQUESTED_CHECKPOINTS[@]}"
do
    case "$requested_checkpoint" in
        old_seed)
            CHECKPOINT_LABELS+=("old_seed")
            CHECKPOINT_PATHS+=("$OLD_CHECKPOINT")
            ;;
        rl_step_30000)
            CHECKPOINT_LABELS+=("rl_step_30000")
            CHECKPOINT_PATHS+=("$RL_CHECKPOINT")
            ;;
        *)
            echo "Unknown checkpoint setting: $requested_checkpoint" >&2
            exit 1
            ;;
    esac
done

lr_tag() {
    echo "$1" | sed 's/-/m/g; s/+//g; s/\./p/g'
}

decode_args=(
    --enc_dec_beam_width "$ENC_DEC_BEAM_WIDTH"
    --enc_dec_length_penalty "$ENC_DEC_LENGTH_PENALTY"
    --enc_dec_eos_bias "$ENC_DEC_EOS_BIAS"
    --enc_dec_repetition_penalty "$ENC_DEC_REPETITION_PENALTY"
    --enc_dec_no_repeat_ngram_size "$ENC_DEC_NO_REPEAT_NGRAM_SIZE"
    --enc_dec_max_generate "$ENC_DEC_MAX_GENERATE"
)
decode_suffix="beam${ENC_DEC_BEAM_WIDTH}_lp$(lr_tag "$ENC_DEC_LENGTH_PENALTY")"
if [ "$ENC_DEC_NO_REPEAT_NGRAM_SIZE" != "0" ]; then
    decode_suffix="${decode_suffix}_ng${ENC_DEC_NO_REPEAT_NGRAM_SIZE}"
fi
if [ "$ENC_DEC_EOS_BIAS" != "0.0" ]; then
    decode_suffix="${decode_suffix}_eos$(lr_tag "$ENC_DEC_EOS_BIAS")"
fi
if [ "$ENC_DEC_REPETITION_PENALTY" != "0.0" ]; then
    decode_suffix="${decode_suffix}_rep$(lr_tag "$ENC_DEC_REPETITION_PENALTY")"
fi
if [ "$ENC_DEC_MAX_GENERATE" != "-1" ]; then
    decode_suffix="${decode_suffix}_max${ENC_DEC_MAX_GENERATE}"
fi

for checkpoint_index in "${!CHECKPOINT_LABELS[@]}"
do
    checkpoint_label="${CHECKPOINT_LABELS[$checkpoint_index]}"
    checkpoint="${CHECKPOINT_PATHS[$checkpoint_index]}"
    if [ ! -f "$checkpoint" ]; then
        echo "Missing checkpoint for ${checkpoint_label}: ${checkpoint}" >&2
        exit 1
    fi

    for dataset in "${DATASETS[@]}"
    do
        for training_mode in "${TRAINING_MODES[@]}"
        do
            for lr in "${LRS[@]}"
            do
                for aug in "${AUGS[@]}"
                do
                    case "$aug" in
                        freq9_width44_time0)
                            aug_kwargs=(spec_augment_freq_mask_param=44 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=9)
                            ;;
                        freq6_width34_time0)
                            aug_kwargs=(spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6)
                            ;;
                        freq3_width24_time0)
                            aug_kwargs=(spec_augment_freq_mask_param=24 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=3)
                            ;;
                        freq1_width12_time0)
                            aug_kwargs=(spec_augment_freq_mask_param=12 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=1)
                            ;;
                        no_aug)
                            aug_kwargs=(spec_augment_freq_mask_param=0 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=0)
                            ;;
                        *)
                            echo "Unknown augmentation setting: $aug" >&2
                            exit 1
                            ;;
                    esac

                    for filter in "${FILTERS[@]}"
                    do
                        case "$filter" in
                            no_filter)
                                filter_args=()
                                run_aug="${aug}"
                                ;;
                            basic_repeat_filter)
                                filter_args=(
                                    --teacher_filter_max_length
                                    --teacher_min_frames_per_token 8
                                    --teacher_filter_max_consecutive_token_repeat
                                    --teacher_max_consecutive_token_repeat 4
                                    --teacher_filter_repeated_words
                                    --teacher_max_consecutive_word_repeat 3
                                )
                                run_aug="${aug}_${filter}"
                                ;;
                            *)
                                echo "Unknown teacher filter setting: $filter" >&2
                                exit 1
                                ;;
                        esac

                        lr_name=$(lr_tag "$lr")
                        run_name="${dataset}-${SPLIT}-${checkpoint_label}-${training_mode}-${decode_suffix}-epoch-${EPOCHS}-lr-${lr_name}-${run_aug}"
                        save_path="${PKL_ROOT}/${run_name}.pkl"
                        log_path="${LOG_ROOT}/${run_name}.log"

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
                            echo "checkpoint_label=${checkpoint_label}"
                            echo "checkpoint=${checkpoint}"
                            echo "save_path=${save_path}"
                            echo "training_mode=${training_mode}"
                            echo "teacher_kl_temperature=${TEACHER_KL_TEMPERATURE}"
                            echo "augmentation=${aug} kwargs=${aug_kwargs[*]}"
                            echo "decode_args=${decode_args[*]}"
                            echo "teacher_filters=${filter} args=${filter_args[*]}"
                        } | tee "$log_path"

                        cmd=(
                            "$PYTHON_BIN" enc_dec_dynamic_eval_test.py
                            --training_mode "$training_mode"
                            --teacher_kl_temperature "$TEACHER_KL_TEMPERATURE"
                            "${filter_args[@]}"
                            -c "$checkpoint"
                            -dfa
                            -epochs "$EPOCHS"
                            -r "$REPEATS"
                            -seq "$SEQ"
                            -o "$OVERLAP"
                            --split "$SPLIT"
                            --dataset "$dataset"
                            -s "$save_path"
                            -log "$log_path"
                            "${decode_args[@]}"
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
        done
    done
done

if [ "$DRY_RUN" != "1" ]; then
    "$PYTHON_BIN" results/enc_dec/rl_step_30000/rob63_rl_self_training_compare/aggregate.py \
        --directory "${PKL_ROOT}" \
        --csv "${RESULTS_ROOT}/summary.csv" \
        --outcome "${RESULTS_ROOT}/OUTCOME.md"
fi
