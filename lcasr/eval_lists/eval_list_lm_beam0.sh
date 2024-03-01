EPOCHS=5
LEARNING_RATE=0.00009
N_FREQ_MASKS=6
FREQ_MASKS_WIDTH=34
LM_TTA_BEAMS=0
REPEAT=3
DEVICE="2"


CUDA_VISIBLE_DEVICES=$DEVICE python run.py -beamsearch -dfa -epochs $EPOCHS -ds -seq 16384 -o 14336 -split dev --dataset tedlium -r $REPEAT -s "./results/lm_beam0/ted_dev.pkl" -kwargs optim_lr=$LEARNING_RATE spec_augment_n_freq_masks=$N_FREQ_MASKS spec_augment_freq_mask_param=$FREQ_MASKS_WIDTH  spec_augment_n_time_masks=0 lm_alpha=0.4016 lm_beta=1.625 lm_eval_beams=20 lm_prune_less_than_val=3.221 lm_tta_beams=$LM_TTA_BEAMS
CUDA_VISIBLE_DEVICES=$DEVICE python run.py -beamsearch -dfa -epochs $EPOCHS -ds -seq 16384 -o 14336 -split test --dataset tedlium -r $REPEAT -s "./results/lm_beam0/ted_test.pkl" -kwargs optim_lr=$LEARNING_RATE spec_augment_n_freq_masks=$N_FREQ_MASKS spec_augment_freq_mask_param=$FREQ_MASKS_WIDTH  spec_augment_n_time_masks=0 lm_alpha=0.4016 lm_beta=1.625 lm_eval_beams=20 lm_prune_less_than_val=3.221 lm_tta_beams=$LM_TTA_BEAMS

CUDA_VISIBLE_DEVICES=$DEVICE python run.py -beamsearch -dfa -epochs $EPOCHS -ds -seq 16384 -o 14336 -split dev --dataset earnings22 -r $REPEAT -s "./results/lm_beam0/earnings22_dev.pkl" -kwargs optim_lr=$LEARNING_RATE spec_augment_n_freq_masks=$N_FREQ_MASKS spec_augment_freq_mask_param=$FREQ_MASKS_WIDTH   spec_augment_n_time_masks=0 lm_alpha=0.4016 lm_beta=1.625 lm_eval_beams=20 lm_prune_less_than_val=3.221 lm_tta_beams=$LM_TTA_BEAMS
CUDA_VISIBLE_DEVICES=$DEVICE python run.py -beamsearch -dfa -epochs $EPOCHS -ds -seq 16384 -o 14336 -split test --dataset earnings22 -r $REPEAT -s "./results/lm_beam0/earnings22_test.pkl" -kwargs optim_lr=$LEARNING_RATE spec_augment_n_freq_masks=$N_FREQ_MASKS spec_augment_freq_mask_param=$FREQ_MASKS_WIDTH  spec_augment_n_time_masks=0 lm_alpha=0.4016 lm_beta=1.625 lm_eval_beams=20 lm_prune_less_than_val=3.221 lm_tta_beams=$LM_TTA_BEAMS

CUDA_VISIBLE_DEVICES=$DEVICE python run.py -beamsearch -dfa -epochs $EPOCHS -ds -seq 16384 -o 14336 -split dev --dataset chime6 -r $REPEAT -s "./results/lm_beam0/chime6_dev.pkl" -kwargs optim_lr=$LEARNING_RATE spec_augment_n_freq_masks=$N_FREQ_MASKS spec_augment_freq_mask_param=$FREQ_MASKS_WIDTH  spec_augment_n_time_masks=0 lm_alpha=0.4016 lm_beta=1.625 lm_eval_beams=20 lm_prune_less_than_val=3.221 lm_tta_beams=$LM_TTA_BEAMS
CUDA_VISIBLE_DEVICES=$DEVICE python run.py -beamsearch -dfa -epochs $EPOCHS -ds -seq 16384 -o 14336 -split test --dataset chime6 -r $REPEAT -s "./results/lm_beam0/chime6_test.pkl" -kwargs optim_lr=$LEARNING_RATE spec_augment_n_freq_masks=$N_FREQ_MASKS spec_augment_freq_mask_param=$FREQ_MASKS_WIDTH   spec_augment_n_time_masks=0 lm_alpha=0.4016 lm_beta=1.625 lm_eval_beams=20 lm_prune_less_than_val=3.221 lm_tta_beams=$LM_TTA_BEAMS

CUDA_VISIBLE_DEVICES=$DEVICE python run.py -beamsearch -dfa -epochs $EPOCHS -ds -seq 16384 -o 14336 -split test --dataset rev16 -r $REPEAT -s "./results/lm_beam0/rev16_test.pkl" -kwargs optim_lr=$LEARNING_RATE spec_augment_n_freq_masks=$N_FREQ_MASKS spec_augment_freq_mask_param=$FREQ_MASKS_WIDTH  spec_augment_n_time_masks=0 lm_alpha=0.4016 lm_beta=1.625 lm_eval_beams=20 lm_prune_less_than_val=3.221 lm_tta_beams=$LM_TTA_BEAMS

