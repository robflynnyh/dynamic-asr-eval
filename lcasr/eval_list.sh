CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 1 -ds -seq 16384 -o 14336 -split dev --dataset tedlium -r 3 -s "./results/online/ted_dev.pkl" -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=10 spec_augment_min_p=0.0 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 online=True
CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 1 -ds -seq 16384 -o 14336 -split test --dataset tedlium -r 3 -s "./results/online/ted_test.pkl" -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=10 spec_augment_min_p=0.0 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 online=True

CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 1 -ds -seq 16384 -o 14336 -split dev --dataset earnings22 -r 3 -s "./results/online/earnings22_dev.pkl" -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=10 spec_augment_min_p=0.0 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 online=True
CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 1 -ds -seq 16384 -o 14336 -split test --dataset earnings22 -r 3 -s "./results/online/earnings22_test.pkl" -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=10 spec_augment_min_p=0.0 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 online=True

CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 1 -ds -seq 16384 -o 14336 -split dev --dataset chime6 -r 3 -s "./results/online/chime6_dev.pkl" -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=10 spec_augment_min_p=0.0 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 online=True
CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 1 -ds -seq 16384 -o 14336 -split test --dataset chime6 -r 3 -s "./results/online/chime6_test.pkl" -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=10 spec_augment_min_p=0.0 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 online=True

CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 1 -ds -seq 16384 -o 14336 -split test --dataset rev16 -r 3 -s "./results/online/rev16_test.pkl" -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=10 spec_augment_min_p=0.0 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 online=True

