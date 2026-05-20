#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU=${GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
EPOCHS=${EPOCHS:-1}
REPEATS=${REPEATS:-1}
SEQ=${SEQ:-2048}
OVERLAP=${OVERLAP:-0}
LR=${LR:-1e-7}
AUG=${AUG:-freq3_width24_time0}
ENC_DEC_BEAM_WIDTH=${ENC_DEC_BEAM_WIDTH:-5}
ENC_DEC_LENGTH_PENALTY=${ENC_DEC_LENGTH_PENALTY:-0.5}
ENC_DEC_EOS_BIAS=${ENC_DEC_EOS_BIAS:-0.0}
ENC_DEC_REPETITION_PENALTY=${ENC_DEC_REPETITION_PENALTY:-0.0}
ENC_DEC_NO_REPEAT_NGRAM_SIZE=${ENC_DEC_NO_REPEAT_NGRAM_SIZE:-0}
ENC_DEC_MAX_GENERATE=${ENC_DEC_MAX_GENERATE:--1}
TEACHER_KL_TEMPERATURE=${TEACHER_KL_TEMPERATURE:-1.0}
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

RESULTS_ROOT=${RESULTS_ROOT:-"./results/enc_dec/rob94_fixed_setting_thesis"}
PKL_ROOT="${RESULTS_ROOT}/pkl"
LOG_ROOT="${RESULTS_ROOT}/logs"
mkdir -p "$PKL_ROOT" "$LOG_ROOT" "$MPLCONFIGDIR"

ENC_DEC_V2_CHECKPOINT=${ENC_DEC_V2_CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt"}
OLD_CHECKPOINT=${OLD_CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt"}
RL_CHECKPOINT=${RL_CHECKPOINT:-"/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt"}

read -r -d '' DEFAULT_RUN_MATRIX <<'EOF' || true
enc_dec_v2 tedlium dev
enc_dec_v2 earnings22 dev
enc_dec_v2 chime6 dev
enc_dec_v2 chime6 test
enc_dec_v2 rev16 test
old_seed tedlium dev
old_seed earnings22 dev
old_seed chime6 dev
rl_step_30000 tedlium dev
rl_step_30000 earnings22 dev
rl_step_30000 chime6 dev
EOF
RUN_MATRIX=${RUN_MATRIX:-$DEFAULT_RUN_MATRIX}

lr_tag() {
    echo "$1" | sed 's/-/m/g; s/+//g; s/\./p/g'
}

checkpoint_path() {
    case "$1" in
        enc_dec_v2) printf '%s\n' "$ENC_DEC_V2_CHECKPOINT" ;;
        old_seed) printf '%s\n' "$OLD_CHECKPOINT" ;;
        rl_step_30000) printf '%s\n' "$RL_CHECKPOINT" ;;
        *)
            echo "Unknown checkpoint setting: $1" >&2
            return 1
            ;;
    esac
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

case "$AUG" in
    freq3_width24_time0)
        aug_kwargs=(spec_augment_freq_mask_param=24 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=3)
        ;;
    *)
        echo "ROB-94 fixed setting expects AUG=freq3_width24_time0, got ${AUG}" >&2
        exit 1
        ;;
esac

lr_name=$(lr_tag "$LR")

while read -r checkpoint_label dataset split
do
    if [ -z "${checkpoint_label:-}" ] || [[ "$checkpoint_label" == \#* ]]; then
        continue
    fi
    if [ "$dataset" = "rev16" ] && [ "$split" != "test" ]; then
        echo "Skipping unavailable rev16/${split}; current rev16 loader exposes test only." >&2
        continue
    fi

    checkpoint=$(checkpoint_path "$checkpoint_label")
    if [ ! -f "$checkpoint" ]; then
        echo "Missing checkpoint for ${checkpoint_label}: ${checkpoint}" >&2
        exit 1
    fi

    run_name="${dataset}-${split}-${checkpoint_label}-teacher_ce-${decode_suffix}-epoch-${EPOCHS}-lr-${lr_name}-${AUG}"
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
        echo "training_mode=teacher_ce"
        echo "learning_rate=${LR}"
        echo "augmentation=${AUG} kwargs=${aug_kwargs[*]}"
        echo "decode_args=${decode_args[*]}"
        echo "teacher_filters=none"
    } | tee "$log_path"

    cmd=(
        "$PYTHON_BIN" enc_dec_dynamic_eval_test.py
        --training_mode teacher_ce
        --teacher_kl_temperature "$TEACHER_KL_TEMPERATURE"
        -c "$checkpoint"
        -dfa
        -epochs "$EPOCHS"
        -r "$REPEATS"
        -seq "$SEQ"
        -o "$OVERLAP"
        --split "$split"
        --dataset "$dataset"
        -s "$save_path"
        -log "$log_path"
        "${decode_args[@]}"
        -kwargs optim_lr="$LR" "${aug_kwargs[@]}"
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
    "$PYTHON_BIN" results/enc_dec/rob94_fixed_setting_thesis/aggregate.py \
        --csv "${RESULTS_ROOT}/summary.csv" \
        --outcome "${RESULTS_ROOT}/OUTCOME.md"
fi
