REPEAT=1
DEVICE="3"
SEQ=16384
OLAP=14336
EPOCHS=(20 40)
DATASETS=("earnings22")
SPLITS=("test")

for dataset in "${DATASETS[@]}"
do
    for split in "${SPLITS[@]}"
    do  
        # print the current file and split
        echo $split
        echo $dataset
        for epoch in "${EPOCHS[@]}"
        do
            echo $epoch
            CUDA_VISIBLE_DEVICES=$DEVICE python run.py -dfa  -epochs $epoch -seq $SEQ -o $OLAP -split $split --dataset $dataset -s "./results/per_epoch_eval/epoch-$epoch-$dataset-$split.pkl" -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6
        done
        echo "done"
    done
done


EPOCHS=(1 2 3 4 5 6 7 8 9 10)
DATASETS=("tedlium" "chime6" "rev16")
SPLITS=("test")

for dataset in "${DATASETS[@]}"
do
    for split in "${SPLITS[@]}"
    do  
        # print the current file and split
        echo $split
        echo $dataset
        for epoch in "${EPOCHS[@]}"
        do
            echo $epoch
            CUDA_VISIBLE_DEVICES=$DEVICE python run.py -dfa  -epochs $epoch -seq $SEQ -o $OLAP -split $split --dataset $dataset -s "./results/per_epoch_eval/epoch-$epoch-$dataset-$split.pkl" -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6
        done
        echo "done"
    done
done
