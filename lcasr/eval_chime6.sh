CUDA_VISIBLE_DEVICES="0" python run.py -dfa -epochs 5 -seq 16384 -o 14336 -split test --dataset chime6 -r 1 -s "./results/freq_mask/chime6_test.pkl" -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0
#CUDA_VISIBLE_DEVICES="0" python run.py -dfa -epochs 5 -seq 16384 -o 14336 -split dev --dataset chime6 -r 3 -s "./results/freq_mask/chime6_dev.pkl" -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34  spec_augment_n_time_masks=0