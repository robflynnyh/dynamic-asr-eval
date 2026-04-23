EPOCHS=(1 5)
DATASET=${DATASET:-earnings22}
SEQ=${SEQ:-16384}
OVERLAP=${OVERLAP:-14336}
LOO_SEQ=${LOO_SEQ:-65536}
LOO_OVERLAP=${LOO_OVERLAP:-57344}
PYTHON_BIN=${PYTHON_BIN:-python3.10}

for epoch in "${EPOCHS[@]}"
do
    echo $epoch
    CUDA_VISIBLE_DEVICES="3" "$PYTHON_BIN" run_within_recording_loo_eval.py -dfa -epochs $epoch -seq $SEQ -o $OVERLAP -loo_s $LOO_SEQ -loo_o $LOO_OVERLAP -split test -d $DATASET -r 1 -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0 -s "./results/within_loo/${DATASET}-loo${LOO_SEQ}_${LOO_OVERLAP}-inner${SEQ}_${OVERLAP}-epoch-$epoch-test.pkl"
    echo "done"
done
