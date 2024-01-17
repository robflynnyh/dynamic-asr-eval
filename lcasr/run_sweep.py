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
        "lr_lr": {
            "min": 1e-9,
            "max": 1e-3,
        },
        "spec_augment_n_time_masks": {
            "min": 0,
            "max": 8,
        },
        "spec_augment_n_freq_masks":{
            "min": 0,
            "max": 6,
        },
        "spec_augment_freq_mask_param":{
            "min": 0,
            "max": 80,
        },
        "spec_augment_min_p":{
            "min": 0,
            "max": 0.3,
        },
        "epochs":{
            "min": 1,
            "max": 4,
        },
    }
}


def launch_agent(args):
    wandb.init()
    args = argparse.Namespace(**{**vars(args), **wandb.config}) # add everything from wandb.config to args

    if args.dataset == 'earnings22':
        run_earnings22.main(args)
    elif args.dataset == 'chime6':
        run_chime6.main(args)
    elif args.dataset == 'tedlium':
        run_tedlium.main(args)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sweep_id', type=str, default='', help='Sweep ID to launch an agent')
    parser.add_argument('--dataset', type=str, default='earnings22', help='Dataset to use', choices=['chime6', 'earnings22', 'tedlium'])
    args = lib.apply_args(parser)

    if args.sweep_id == '':
        sweep_id = wandb.sweep(sweep_config, project='dynamic-eval-sweep')
        print(f'Sweep ID: {sweep_id}')
    else:
        sweep_id = args.sweep_id
    
    wandb.agent(sweep_id, partial(launch_agent, args))

