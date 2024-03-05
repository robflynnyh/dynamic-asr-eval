EPOCHS=(1 2 3 4 5)

CUDA_VISIBLE_DEVICES="3" python run_seq_eval.py -dfa -epochs 0 -seq 16384 -o 14336 -nsti_s -1 -nsti_o 0  --dataset earnings22 -r 1  -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0 -s "./results/seqlens/earnings22-baseline.pkl"

for epoch in "${EPOCHS[@]}"
do
    echo $epoch
    CUDA_VISIBLE_DEVICES="3" python run_seq_eval.py -dfa -epochs $epoch -seq 16384 -o 0 -nsti_s 16384 -nsti_o 14336  --dataset earnings22 -r 1  -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0 -s "./results/seqlens/earnings22-16384-epoch-$epoch-test.pkl"
    CUDA_VISIBLE_DEVICES="3" python run_seq_eval.py -dfa -epochs $epoch -seq 16384 -o 14336 -nsti_s 32768 -nsti_o 28672  --dataset earnings22 -r 1  -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0 -s "./results/seqlens/earnings22-32768-epoch-$epoch-test.pkl"
    CUDA_VISIBLE_DEVICES="3" python run_seq_eval.py -dfa -epochs $epoch -seq 16384 -o 14336 -nsti_s 65536 -nsti_o 57344  --dataset earnings22 -r 1  -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0 -s "./results/seqlens/earnings22-65536-epoch-$epoch-test.pkl"
    CUDA_VISIBLE_DEVICES="3" python run_seq_eval.py -dfa -epochs $epoch -seq 16384 -o 14336 -nsti_s 131072 -nsti_o 114688  --dataset earnings22 -r 1  -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0 -s "./results/seqlens/earnings22-131072-epoch-$epoch-test.pkl"
    CUDA_VISIBLE_DEVICES="3" python run_seq_eval.py -dfa -epochs $epoch -seq 16384 -o 14336 -nsti_s 262144 -nsti_o 229376 --dataset earnings22 -r 1  -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0 -s "./results/seqlens/earnings22-262144-epoch-$epoch-test.pkl"
    CUDA_VISIBLE_DEVICES="3" python run_seq_eval.py -dfa -epochs $epoch -seq 16384 -o 14336 -nsti_s 360000 -nsti_o 315000  --dataset earnings22 -r 1  -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0 -s "./results/seqlens/earnings22-360000-epoch-$epoch-test.pkl"
    echo "done"
done






# 16384
# 32768
# 65536
# 131072
# 262144
# 360000 # 1 hour