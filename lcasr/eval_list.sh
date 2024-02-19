CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 7 -ds -seq 16384 -o 14336 -split dev --dataset tedlium -r 3 -s "./results/ordered/ted_dev.pkl" -kwargs optim_lr=0.00001 spec_augment_freq_mask_param=50 spec_augment_min_p=0.0 spec_augment_n_freq_masks=4  spec_augment_n_time_masks=0 
CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 7 -ds -seq 16384 -o 14336 -split test --dataset tedlium -r 3 -s "./results/ordered/ted_test.pkl" -kwargs optim_lr=0.00001 spec_augment_freq_mask_param=50 spec_augment_min_p=0.0 spec_augment_n_freq_masks=4  spec_augment_n_time_masks=0 

CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 7 -ds -seq 16384 -o 14336 -split dev --dataset earnings22 -r 3 -s "./results/ordered/earnings22_dev.pkl" -kwargs optim_lr=0.00001 spec_augment_freq_mask_param=50 spec_augment_min_p=0.0 spec_augment_n_freq_masks=4  spec_augment_n_time_masks=0 
CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 7 -ds -seq 16384 -o 14336 -split test --dataset earnings22 -r 3 -s "./results/ordered/earnings22_test.pkl" -kwargs optim_lr=0.00001 spec_augment_freq_mask_param=50 spec_augment_min_p=0.0 spec_augment_n_freq_masks=4  spec_augment_n_time_masks=0 

CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 7 -ds -seq 16384 -o 14336 -split test --dataset rev16 -r 3 -s "./results/ordered/rev16_test.pkl" -kwargs optim_lr=0.00001 spec_augment_freq_mask_param=50 spec_augment_min_p=0.0 spec_augment_n_freq_masks=4  spec_augment_n_time_masks=0 

CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 7 -ds -seq 16384 -o 14336 -split dev --dataset chime6 -r 3 -s "./results/ordered/chime6_dev.pkl" -kwargs optim_lr=0.00001 spec_augment_freq_mask_param=50 spec_augment_min_p=0.0 spec_augment_n_freq_masks=4  spec_augment_n_time_masks=0 
CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 7 -ds -seq 16384 -o 14336 -split test --dataset chime6 -r 3 -s "./results/ordered/chime6_test.pkl" -kwargs optim_lr=0.00001 spec_augment_freq_mask_param=50 spec_augment_min_p=0.0 spec_augment_n_freq_masks=4  spec_augment_n_time_masks=0 

