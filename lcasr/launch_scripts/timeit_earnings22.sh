CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 1 -ds -seq 16384 -o 14336 -split test --dataset earnings22 -r 1 -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=10 spec_augment_min_p=0.0 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 online=True print_runtimes=True 

CUDA_VISIBLE_DEVICES="3" python run.py -dfa -epochs 1 -awmc -ds -seq 16384 -o 14336 -split test --dataset earnings22 -r 1 -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=10 spec_augment_min_p=0.0 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 online=True print_runtimes=True 


415990
0.865405418
95.76766729354858 * 0.865405418 = 82.877858145 - RTF = 95.76766729354858/4159.90 =  0.023

403.6457395553589
415990
RTF = 403.6457395553589/4159.90 = 0.097


92.31600046157837
18.04189705848694