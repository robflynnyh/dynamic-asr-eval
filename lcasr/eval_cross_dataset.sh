EPOCHS=(0 1 2 3 4 5)

for epoch in "${EPOCHS[@]}"
do
    echo $epoch
    CUDA_VISIBLE_DEVICES="0" python run_in_dataset_eval.py -dfa -epochs $epoch -seq 16384 -o 14336 -split test -d earnings22 -r 1 -s "./results/indataset/earnings22-$epoch-test.pkl" -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0
    echo "done"
done


EPOCHS=(1 2 3 4 5)

for epoch in "${EPOCHS[@]}"
do
    echo $epoch
    CUDA_VISIBLE_DEVICES="0" python run_cross_dataset_eval.py -dfa -epochs $epoch -seq 16384 -o 14336 -split test -d earnings22 -d2 tedlium -r 1 -s "./results/crossdataset/earnings_tedlium_epoch-$epoch-test.pkl" -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0
    echo "done"
done