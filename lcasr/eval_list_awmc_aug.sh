EPOCHS=5
LEARNING_RATE=0.00001
N_FREQ_MASKS=5
FREQ_MASKS_WIDTH=23
DEVICE="2"


CUDA_VISIBLE_DEVICES=$DEVICE python run.py -awmc -dfa -epochs $EPOCHS -ds -seq 16384 -o 14336 -split dev --dataset tedlium -r 3 -s "./results/awmc_aug/ted_dev.pkl" -kwargs optim_lr=$LEARNING_RATE spec_augment_n_freq_masks=$N_FREQ_MASKS spec_augment_freq_mask_param=$FREQ_MASKS_WIDTH  spec_augment_n_time_masks=0 
CUDA_VISIBLE_DEVICES=$DEVICE python run.py -awmc -dfa -epochs $EPOCHS -ds -seq 16384 -o 14336 -split test --dataset tedlium -r 3 -s "./results/awmc_aug/ted_test.pkl" -kwargs optim_lr=$LEARNING_RATE spec_augment_n_freq_masks=$N_FREQ_MASKS spec_augment_freq_mask_param=$FREQ_MASKS_WIDTH  spec_augment_n_time_masks=0 

CUDA_VISIBLE_DEVICES=$DEVICE python run.py -awmc -dfa -epochs $EPOCHS -ds -seq 16384 -o 14336 -split dev --dataset earnings22 -r 3 -s "./results/awmc_aug/earnings22_dev.pkl" -kwargs optim_lr=$LEARNING_RATE spec_augment_n_freq_masks=$N_FREQ_MASKS spec_augment_freq_mask_param=$FREQ_MASKS_WIDTH   spec_augment_n_time_masks=0 
CUDA_VISIBLE_DEVICES=$DEVICE python run.py -awmc -dfa -epochs $EPOCHS -ds -seq 16384 -o 14336 -split test --dataset earnings22 -r 3 -s "./results/awmc_aug/earnings22_test.pkl" -kwargs optim_lr=$LEARNING_RATE spec_augment_n_freq_masks=$N_FREQ_MASKS spec_augment_freq_mask_param=$FREQ_MASKS_WIDTH  spec_augment_n_time_masks=0 

CUDA_VISIBLE_DEVICES=$DEVICE python run.py -awmc -dfa -epochs $EPOCHS -ds -seq 16384 -o 14336 -split dev --dataset chime6 -r 3 -s "./results/awmc_aug/chime6_dev.pkl" -kwargs optim_lr=$LEARNING_RATE spec_augment_n_freq_masks=$N_FREQ_MASKS spec_augment_freq_mask_param=$FREQ_MASKS_WIDTH  spec_augment_n_time_masks=0 
CUDA_VISIBLE_DEVICES=$DEVICE python run.py -awmc -dfa -epochs $EPOCHS -ds -seq 16384 -o 14336 -split test --dataset chime6 -r 3 -s "./results/awmc_aug/chime6_test.pkl" -kwargs optim_lr=$LEARNING_RATE spec_augment_n_freq_masks=$N_FREQ_MASKS spec_augment_freq_mask_param=$FREQ_MASKS_WIDTH   spec_augment_n_time_masks=0 

CUDA_VISIBLE_DEVICES=$DEVICE python run.py -awmc -dfa -epochs $EPOCHS -ds -seq 16384 -o 14336 -split test --dataset rev16 -r 3 -s "./results/awmc_aug/rev16_test.pkl" -kwargs optim_lr=$LEARNING_RATE spec_augment_n_freq_masks=$N_FREQ_MASKS spec_augment_freq_mask_param=$FREQ_MASKS_WIDTH  spec_augment_n_time_masks=0 

