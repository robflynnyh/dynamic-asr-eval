import argparse
import wandb
import lib
from functools import partial
from omegaconf import OmegaConf
from run import main



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
    parser.add_argument('-d','--dataset', type=str, default='tedlium', help='Dataset to use', choices=['chime6', 'earnings22', 'tedlium'])
    parser.add_argument('--repeats', '-r', type=int, default=1, help='Number of times to repeat the evaluation')
    parser.add_argument('-sc', '--sweep_config', required=True, help="Path to sweep config file")
   
    args = lib.apply_args(parser)
    if args.split == 'test': print(f'Split set to test set, but sweeps are only performed on dev, setting split to dev.')
    args.split = 'dev' # only run sweep on dev set

    sweep_config = OmegaConf.load(args.sweep_config)
    sweep_config = OmegaConf.to_container(sweep_config, resolve=True)
    
    if args.sweep_id == '':
        sweep_id = wandb.sweep(sweep_config, project='dynamic-eval-sweep')
        print(f'Sweep ID: {sweep_id}')
    else:
        sweep_id = args.sweep_id
    
    wandb.agent(sweep_id, partial(launch_agent, args), project='dynamic-eval-sweep')

#CUDA_VISIBLE_DEVICES="3" python run_sweep.py -dfa -split 'dev' -seq 16384 -o 14336 -sweep_id x07cbzxv
    #2o6bcj0f
#CUDA_VISIBLE_DEVICES="3" python run_sweep.py -dfa -seq 16384 -o 14336 -split dev -sweep_id 2o6bcj0f