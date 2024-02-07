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
            "values": [1e-06, 1e-05, 5e-05, 9e-05, 0.0001, 0.0002, 0.0005],
        },
        "spec_augment_n_time_masks": {
            "values": [0, 5, 10],
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
            "values": [0.0, 0.01, 0.05],
        },
        "epochs":{
            "min": 2,
            "max": 9,
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

#CUDA_VISIBLE_DEVICES="3" python run_sweep.py -dfa -split 'dev' -seq 16384 -o 14336 -sweep_id x07cbzxv