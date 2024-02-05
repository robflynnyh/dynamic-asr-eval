import argparse
import wandb
import lib
from functools import partial

from earnings22 import run as run_earnings22
from chime6 import run as run_chime6
from tedlium import run as run_tedlium

sweep_config = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {
        'name': 'WER',
        'goal': 'minimize'
    },
    "parameters": {
        "optim_lr": {
            "values": [1e-9, 1e-8, 5e-8, 1e-7, 5e-7, 9e-7, 1e-6, 5e-6, 8e-6, 5e-5]
        },
        "spec_augment_n_time_masks": {
            "values": [0],
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
            "values": [0.0],
        },
        "epochs":{
            "min": 2,
            "max": 8,
        },
    }
}


def launch_agent(args):
    wandb.init()
    
    args_dict = vars(args)
    
    for k in wandb.config.keys():
        args_dict[k] = wandb.config[k]
        
    args = argparse.Namespace(**args_dict)
    
    if args.dataset == 'earnings22':
        wer = run_earnings22.main(args)
    elif args.dataset == 'chime6':
        wer = run_chime6.main(args)
    elif args.dataset == 'tedlium':
        wer = run_tedlium.main(args)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    
    if wer == None:
        raise ValueError('WER is None - something went wrong')
    
    wandb.log({'WER': wer})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sweep_id','-sweep_id', type=str, default='', help='Sweep ID to launch an agent')
    parser.add_argument('-dataset','--dataset', type=str, default='earnings22', help='Dataset to use', choices=['chime6', 'earnings22', 'tedlium'])
    args = lib.apply_args(parser)

    if args.sweep_id == '':
        sweep_id = wandb.sweep(sweep_config, project='dynamic-eval-sweep')
        print(f'Sweep ID: {sweep_id}')
    else:
        sweep_id = args.sweep_id
    
    wandb.agent(sweep_id, partial(launch_agent, args), project='dynamic-eval-sweep')

