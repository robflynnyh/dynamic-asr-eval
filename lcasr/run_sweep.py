import argparse
import wandb
import lib
from functools import partial

from run import main

sweep_config = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {
        'name': 'WER',
        'goal': 'minimize'
    },
    "parameters": {
        "optim_lr": {
            "values": [1e-6,1e-05, 5e-05, 9e-05, 0.0001, 0.0002],
        },
        "spec_augment_n_time_masks": {
            "values": [0, 3, 10],
        },
        "spec_augment_n_freq_masks":{
            "min": 0,
            "max": 15,
        },
        "spec_augment_freq_mask_param":{
            "min": 0,
            "max": 80,
        },
        "spec_augment_min_p":{
            "values": [0.0, 0.1],
        },
        "epochs":{
            "min": 1,
            "max": 9,
        },
    }
}


def launch_agent(args):
    wandb.init()
    
    args_dict = vars(args)
    
    for k in wandb.config.keys():
        args_dict[k] = wandb.config[k]
    args_dict['save_path'] = ''
        
    args = argparse.Namespace(**args_dict)
    
    wer = main(args)
    
    if wer == None:
        raise ValueError('WER is None - something went wrong')
    
    wandb.log({'WER': wer})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sweep_id','-sweep_id', type=str, default='', help='Sweep ID to launch an agent')
    parser.add_argument('-d','--dataset', type=str, default='earnings22', help='Dataset to use', choices=['chime6', 'earnings22', 'tedlium'])
    parser.add_argument('--repeats', '-r', type=int, default=1, help='Number of times to repeat the evaluation')
   
    args = lib.apply_args(parser)

    if args.sweep_id == '':
        sweep_id = wandb.sweep(sweep_config, project='dynamic-eval-sweep')
        print(f'Sweep ID: {sweep_id}')
    else:
        sweep_id = args.sweep_id
    
    wandb.agent(sweep_id, partial(launch_agent, args), project='dynamic-eval-sweep')

#CUDA_VISIBLE_DEVICES="3" python run_sweep.py -dfa -split 'dev' -seq 16384 -o 14336 -sweep_id x07cbzxv
    #2o6bcj0f
#CUDA_VISIBLE_DEVICES="3" python run_sweep.py -dfa -seq 16384 -o 14336 -split dev -sweep_id 2o6bcj0f