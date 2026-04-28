import torch, argparse, lcasr, os, re, json, pickle, time
from tqdm import tqdm
from typing import Tuple
from lcasr.utils.audio_tools import processing_chain
from lcasr.eval.utils import fetch_logits, decode_beams_lm
from lcasr.utils.general import load_model, get_model_class
from lcasr.eval.wer import word_error_rate_detail 
#from lcasr.eval.dynamic_eval import dynamic_eval
from whisper.normalizers import EnglishTextNormalizer

normalize = EnglishTextNormalizer()

import sys
import os.path

import lib
from lib import enc_dec_dynamic_eval
from enc_dec_teacher_filters import add_enc_dec_teacher_filter_args

from earnings22.run import get_text_and_audio as get_text_and_audio_earnings22
from chime6.run import get_text_and_audio as get_text_and_audio_chime6
from tedlium.run import get_text_and_audio as get_text_and_audio_tedlium
from rev16.run import get_text_and_audio as get_text_and_audio_rev16

datasets_functions = {
    'earnings22': get_text_and_audio_earnings22,
    'chime6': get_text_and_audio_chime6,
    'tedlium': get_text_and_audio_tedlium,
    'rev16': get_text_and_audio_rev16
}



def main(args):
    assert args.split in ['test', 'dev'], f'Split must be either test or dev (got {args.split})'
    if args.dataset == 'rev16': assert args.split == 'test', 'Split must be test for rev16'
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_config = checkpoint['config']
    args.config = model_config

    if args.disable_flash_attention:
        args.config.model.flash_attn = False

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, model_class=get_model_class(args.config), vocab_size=len(tokenizer))
    tparams = model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()

    data = datasets_functions[args.dataset](args.split)

    avg_wers = []

    for repeat in range(args.repeats):
        all_texts, all_golds = [],[]
        elapsed_times = []
    
        for rec in tqdm(range(len(data)), total=len(data)):
            print(f'Processing {rec+1}/{len(data)}')
            
            print('\n-------\n'+data[rec]['id']+'\n-------\n')
        
            audio_spec, gold_text = data[rec]['process_fn'](data[rec])
            
            stime = time.time()
            model_out = enc_dec_dynamic_eval(
                args = args,
                model = model,
                spec = audio_spec,
                seq_len = args.seq_len,
                overlap = 0,
                tokenizer = tokenizer,
                use_tqdm = True
            )
            etime = time.time()
            elapsed_times.append(etime - stime)

            out = normalize(model_out).lower()
            
            print(gold_text, '\n', out, '\n\n')
            
            all_texts.append(out)
            all_golds.append(gold_text)

            if args.breaks:
                break
            

        wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

        print(f'WER: {wer}')

        if args.log != '':
            with open(args.log, 'a') as f:
                f.write(f'{args.checkpoint}\t overlap: {args.overlap}\t seq_len: {args.seq_len}\t WER: {wer}\n')

        if args.save_path != '':
            save_data = {
                'wer': wer,
                'words': words,
                'ins_rate': ins_rate,
                'del_rate': del_rate,
                'sub_rate': sub_rate,
                'model_output': all_texts,
                'gold': all_golds,
                'elapsed_times': elapsed_times,
                'args_dict': vars(args),
                'repeat': f'{repeat + 1}/{args.repeats}'
            }
            save_path = args.save_path
            if save_path.endswith('.pkl'):
                save_path = save_path.replace('.pkl', f'_{repeat + 1}.pkl')
            else:
                save_path = save_path + f'_{repeat + 1}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            print(f'Saved to {save_path}')

        avg_wers.append(wer)

    avg = sum(avg_wers) / len(avg_wers)
    print(f'Average WER: {avg}')
    return avg
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='earnings22', choices=datasets_functions.keys())
    parser.add_argument('--repeats', '-r', type=int, default=1, help='Number of times to repeat the evaluation')
    parser.add_argument('--save_path', '-s', type=str, default='', help='path to save')
    parser.add_argument('--breaks', action='store_true', help='Break after first sample (for debugging)')
    parser.add_argument('--training_mode', type=str, default='grpo',
                        choices=['grpo', 'maxrl', 'teacher_ce'],
                        help='Update rule for enc-dec TTA after the teacher-filter gate. '
                             'grpo: REINFORCE-style with per-rollout reward centering (default). '
                             'maxrl: MaxRL (Tajwar et al. 2026, arXiv:2602.02710); binarises rewards via --maxrl_success_threshold. '
                             'teacher_ce: no RL — supervised cross-entropy on the (filtered) teacher prediction.')
    parser.add_argument('--maxrl_success_threshold', type=float, default=0.9,
                        help='Continuous-reward threshold for binarising rollouts as success/failure under --training_mode maxrl. Default 0.9 ~= error<0.1 under the calc_rewards mean.')
    parser.add_argument('--grpo_normalize_std', action=argparse.BooleanOptionalAction, default=True,
                        help='Normalize GRPO advantages by group reward std. Enabled by default; use --no-grpo_normalize_std to disable.')
    add_enc_dec_teacher_filter_args(parser)
    args = lib.apply_args(parser)
    main(args)
    

#python run.py -d earnings22 -r 3 -dfa -epochs 5 -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=34 spec_augment_min_p=0.1879883950862319 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6

#CUDA_VISIBLE_DEVICES="3" python enc_dec_dynamic_eval_test.py -dfa -d tedlium -split dev -seq 2048 -o 0 -c /store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec/3e3/step_105360.pt 
