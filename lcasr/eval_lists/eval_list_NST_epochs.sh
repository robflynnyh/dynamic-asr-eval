EPOCHS=0
REPEAT=1
DEVICE="1"

CHECKPOINTS=("5" "10" "20" "40" "80" "100" "101" "120" "140")
SPLITS=("dev" "test")

for file in "${CHECKPOINTS[@]}"
do
    for split in "${SPLITS[@]}"
    do  
        # print the current file and split
        echo "/store/store5/data/acp21rjf_checkpoints/earningsNST/$file.pt"
        echo $split
        CUDA_VISIBLE_DEVICES=$DEVICE python run.py -c /store/store5/data/acp21rjf_checkpoints/earningsNST/$file.pt -dfa -r $REPEAT -epochs $EPOCHS -ds -seq 16384 -o 14336 -split $split --dataset earnings22 -s "./results/earningsNST/earnings22_$file-$split.pkl"
        echo "done"
    done
done
