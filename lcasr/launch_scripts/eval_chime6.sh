CUDA_VISIBLE_DEVICES="1" \
    python run.py -dfa -epochs 1 -seq 16384 -o 14336 \
    -split test --dataset earnings22 -r 1 \
    -kwargs optim_lr=9e-5 spec_augment_n_freq_masks=6 \
    spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0 \
    shuffle=True entropy_augmentation_enabled=True 


#CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 0 -seq 16384 -o 14336 -split test --dataset chime6 -r 1 -s "./results/chime6_test_baseline.pkl" -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0
#CUDA_VISIBLE_DEVICES="0" python run.py -dfa -epochs 5 -seq 16384 -o 14336 -split dev --dataset chime6 -r 3 -s "./results/freq_mask/chime6_dev.pkl" -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0