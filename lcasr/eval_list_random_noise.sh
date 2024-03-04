EPOCHS=5
REPEAT=3
DEVICE="2"
SEQ=16384
OLAP=14336

#DATASETS=("tedlium" "earnings22" "chime6")
DATASETS=("chime6")
SPLITS=("dev" "test")

for dataset in "${DATASETS[@]}"
do
    for split in "${SPLITS[@]}"
    do  
        # print the current file and split
        echo $split
        echo $dataset
        CUDA_VISIBLE_DEVICES=$DEVICE python run.py -dfa  -epochs $EPOCHS -seq $SEQ -o $OLAP -split $split --dataset $dataset -s "./results/random_noise/$dataset-$split.pkl" -kwargs optim_lr=0.0001 random_noise=0.32282279559339133
        echo "done"
    done
done

CUDA_VISIBLE_DEVICES=$DEVICE python run.py -dfa  -epochs $EPOCHS -seq $SEQ -o $OLAP -split test --dataset rev16 -s "./results/random_noise/rev16-test.pkl" -kwargs optim_lr=0.0001 random_noise=0.32282279559339133