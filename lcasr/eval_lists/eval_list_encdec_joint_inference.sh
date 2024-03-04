EPOCHS=0
REPEAT=1
DEVICE="2"
SEQ=2048
OLAP=0

DATASETS=("tedlium" "earnings22" "chime6")
SPLITS=("dev" "test")

for dataset in "${DATASETS[@]}"
do
    for split in "${SPLITS[@]}"
    do  
        # print the current file and split
        echo $split
        echo $dataset
        CUDA_VISIBLE_DEVICES=$DEVICE python enc_dec_inference_test.py -mode "joint" -c /store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt -dfa  -epochs $EPOCHS -ds -seq $SEQ -o $OLAP -split $split --dataset $dataset -s "./results/enc_dec_joint_eval/$dataset-$split.pkl"
        echo "done"
    done
done

CUDA_VISIBLE_DEVICES=$DEVICE python enc_dec_inference_test.py -mode "joint" -c /store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt -dfa  -epochs $EPOCHS -ds -seq $SEQ -o $OLAP -split test --dataset rev16 -s "./results/enc_dec_joint_eval/rev16-test.pkl"
