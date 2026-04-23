EPOCHS=(1 5)
PYTHON_BIN=${PYTHON_BIN:-python3.10}

for epoch in "${EPOCHS[@]}"
do
    echo $epoch
    CUDA_VISIBLE_DEVICES="3" "$PYTHON_BIN" run_cross_dataset_eval.py -dfa -epochs $epoch -seq 16384 -o 14336 -split test -d earnings22 -d2 tedlium -r 3 -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0 -s "./results/crossdataset/earnings22_tedlium-epoch-$epoch-test.pkl"
    echo "done"
done
