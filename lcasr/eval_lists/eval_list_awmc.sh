# CUDA_VISIBLE_DEVICES="1" python run.py -awmc -dfa -epochs 1 -ds -seq 16384 -o 14336 -split dev --dataset tedlium -r 3 -s "./results/awmc/ted_dev.pkl" -kwargs optim_lr=0.0002 spec_augment_n_freq_masks=0  spec_augment_n_time_masks=0 
CUDA_VISIBLE_DEVICES="1" python run.py -awmc -dfa -epochs 1 -ds -seq 16384 -o 14336 -split test --dataset tedlium -r 3 -s "./results/awmc/ted_test.pkl" -kwargs optim_lr=0.0002 spec_augment_n_freq_masks=0  spec_augment_n_time_masks=0 

# CUDA_VISIBLE_DEVICES="1" python run.py -awmc -dfa -epochs 1 -ds -seq 16384 -o 14336 -split dev --dataset earnings22 -r 3 -s "./results/awmc/earnings22_dev.pkl" -kwargs optim_lr=0.0002 spec_augment_n_freq_masks=0  spec_augment_n_time_masks=0 
# CUDA_VISIBLE_DEVICES="1" python run.py -awmc -dfa -epochs 1 -ds -seq 16384 -o 14336 -split test --dataset earnings22 -r 3 -s "./results/awmc/earnings22_test.pkl" -kwargs optim_lr=0.0002 spec_augment_n_freq_masks=0  spec_augment_n_time_masks=0 

CUDA_VISIBLE_DEVICES="1" python run.py -awmc -dfa -epochs 1 -ds -seq 16384 -o 14336 -split dev --dataset chime6 -r 3 -s "./results/awmc/chime6_dev.pkl" -kwargs optim_lr=0.0002 spec_augment_n_freq_masks=0  spec_augment_n_time_masks=0 
CUDA_VISIBLE_DEVICES="1" python run.py -awmc -dfa -epochs 1 -ds -seq 16384 -o 14336 -split test --dataset chime6 -r 3 -s "./results/awmc/chime6_test.pkl" -kwargs optim_lr=0.0002 spec_augment_n_freq_masks=0  spec_augment_n_time_masks=0 

# CUDA_VISIBLE_DEVICES="1" python run.py -awmc -dfa -epochs 1 -ds -seq 16384 -o 14336 -split test --dataset rev16 -r 3 -s "./results/awmc/rev16_test.pkl" -kwargs optim_lr=0.0002 spec_augment_n_freq_masks=0  spec_augment_n_time_masks=0 

