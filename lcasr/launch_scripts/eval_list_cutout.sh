EPOCHS=7
REPEAT=3
DEVICE="1"
SEQ=16384
OLAP=14336

DATASETS=("tedlium" "earnings22" "chime6")
SPLITS=("dev" "test")


for dataset in "${DATASETS[@]}"
do
    for split in "${SPLITS[@]}"
    do  
        # print the current file and split
        echo $split
        echo $dataset
        CUDA_VISIBLE_DEVICES=$DEVICE python run.py -dfa -r $REPEAT  -epochs $EPOCHS -seq $SEQ -o $OLAP -split $split --dataset $dataset -s "./results/cutout/$dataset-$split.pkl" -kwargs optim_lr=0.00005 cutout_value="'mean_recording'" cutout_max_height=41 cutout_max_width=792 cutout_num_rectangles=205
        echo "done"
    done
done


CUDA_VISIBLE_DEVICES=$DEVICE python run.py -dfa -r $REPEAT  -epochs $EPOCHS -seq $SEQ -o $OLAP -split test --dataset rev16 -s "./results/cutout/rev16-test.pkl" -kwargs optim_lr=0.00005 cutout_value="'mean_recording'" cutout_max_height=41 cutout_max_width=792 cutout_num_rectangles=205