#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU=${GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
CHECKPOINT=${CHECKPOINT:-"/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0/step_323484.pt"}
EPOCHS=${EPOCHS:-1}
REPEATS=${REPEATS:-1}
SEQ=${SEQ:-2048}
OVERLAP=${OVERLAP:-0}
LR=${LR:-1e-7}
ENC_DEC_BEAM_WIDTH=${ENC_DEC_BEAM_WIDTH:-5}
ENC_DEC_LENGTH_PENALTY=${ENC_DEC_LENGTH_PENALTY:-0.5}
ENC_DEC_EOS_BIAS=${ENC_DEC_EOS_BIAS:-0.0}
ENC_DEC_REPETITION_PENALTY=${ENC_DEC_REPETITION_PENALTY:-0.0}
ENC_DEC_NO_REPEAT_NGRAM_SIZE=${ENC_DEC_NO_REPEAT_NGRAM_SIZE:-0}
ENC_DEC_MAX_GENERATE=${ENC_DEC_MAX_GENERATE:--1}
TEACHER_KL_TEMPERATURE=${TEACHER_KL_TEMPERATURE:-1.0}
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
BREAKS=${BREAKS:-0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"../symphony/.scratch/ROB-122/matplotlib"}

RESULTS_ROOT=${RESULTS_ROOT:-"./results/enc_dec/rob81_floras50_finetune_eval"}
PKL_ROOT="${RESULTS_ROOT}/pkl"
LOG_ROOT="${RESULTS_ROOT}/logs"
mkdir -p "$PKL_ROOT" "$LOG_ROOT" "$MPLCONFIGDIR"

read -r -d '' DEFAULT_RUN_MATRIX <<'EOF' || true
tedlium dev
tedlium test
earnings22 dev
earnings22 test
chime6 dev
chime6 test
rev16 test
EOF
RUN_MATRIX=${RUN_MATRIX:-$DEFAULT_RUN_MATRIX}

RUN_MODES_STR=${RUN_MODES:-"no_adapt teacher_ce"}
read -r -a RUN_MODES_ARRAY <<< "$RUN_MODES_STR"

tag_value() {
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
    echo "Missing ROB-81 checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

extra_runner_args=()
if [ "$BREAKS" = "1" ]; then
    extra_runner_args+=(--breaks)
fi

while read -r dataset split
do
    if [ -z "${dataset:-}" ] || [[ "$dataset" == \#* ]]; then
        continue
    fi

    for run_mode in "${RUN_MODES_ARRAY[@]}"
    do
        case "$run_mode" in
            no_adapt)
                mode_label="no_adapt"
                run_epochs=0
                lr_name="none"
                aug_name="no_aug"
                optim_lr="0.0"
                aug_kwargs=(spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=0)
                ;;
            teacher_ce)
                mode_label="teacher_ce"
                run_epochs="$EPOCHS"
                lr_name=$(tag_value "$LR")
                aug_name="freq3_width24_time0"
                optim_lr="$LR"
                aug_kwargs=(spec_augment_freq_mask_param=24 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=3)
                ;;
            *)
                echo "Unknown run mode: ${run_mode}" >&2
                exit 1
                ;;
        esac

        run_name="${dataset}-${split}-rob81_floras50-${mode_label}-${decode_suffix}-epoch-${run_epochs}-lr-${lr_name}-${aug_name}"
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
            echo "checkpoint_label=rob81_floras50"
            echo "checkpoint=${CHECKPOINT}"
            echo "save_path=${save_path}"
            echo "dataset=${dataset}"
            echo "split=${split}"
            echo "training_mode=${mode_label}"
            echo "epochs=${run_epochs}"
            echo "learning_rate=${optim_lr}"
            echo "augmentation=${aug_name} kwargs=${aug_kwargs[*]}"
            echo "decode_args=${decode_args[*]}"
            echo "teacher_filters=none"
            echo "breaks=${BREAKS}"
        } | tee "$log_path"

        cmd=(
            "$PYTHON_BIN" enc_dec_dynamic_eval_test.py
            --training_mode teacher_ce
            --teacher_kl_temperature "$TEACHER_KL_TEMPERATURE"
            -c "$CHECKPOINT"
            -dfa
            -epochs "$run_epochs"
            -r "$REPEATS"
            -seq "$SEQ"
            -o "$OVERLAP"
            --split "$split"
            --dataset "$dataset"
            -s "$save_path"
            -log "$log_path"
            "${decode_args[@]}"
            "${extra_runner_args[@]}"
            -kwargs optim_lr="$optim_lr" "${aug_kwargs[@]}"
        )

        if [ "$DRY_RUN" = "1" ]; then
            printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU" | tee -a "$log_path"
            printf ' %q' "${cmd[@]}" | tee -a "$log_path"
            printf '\n' | tee -a "$log_path"
        else
            CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" 2>&1 | tee -a "$log_path"
        fi
    done
done <<< "$RUN_MATRIX"

if [ "$DRY_RUN" != "1" ]; then
    "$PYTHON_BIN" results/enc_dec/rob81_floras50_finetune_eval/aggregate.py \
        --csv "${RESULTS_ROOT}/summary.csv" \
        --outcome "${RESULTS_ROOT}/OUTCOME.md"
fi
