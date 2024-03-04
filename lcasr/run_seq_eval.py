import torch, argparse, lcasr, os, re, json
from tqdm import tqdm
from typing import Tuple
from lcasr.utils.audio_tools import processing_chain
from lcasr.eval.utils import fetch_logits, decode_beams_lm
from lcasr.utils.general import load_model
from pyctcdecode import build_ctcdecoder
from lcasr.eval.wer import word_error_rate_detail 
#from lcasr.eval.dynamic_eval import dynamic_eval
from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()
import pickle
import sys
import os.path

import lib
import time
from lib import dynamic_eval, AWMC

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
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint['config']
    args.config = model_config

    if args.disable_flash_attention:
        args.config.model.flash_attn = False

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size())
    tparams = model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()

    vocab = [tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())] + [""]
    decoder = build_ctcdecoder(vocab, kenlm_model_path=None, alpha=None, beta=None)

    data = datasets_functions[args.dataset](args.split)
    
    beamsearch = None
    if args.beamsearch: 
        beamsearch = lib.load_beamsearch(
            path = lib.paths.checkpoints.lm,
            alpha = args.__dict__.get('lm_alpha', 0.45),
            beta = args.__dict__.get('lm_beta', 1.53),
            prune_less_than_val = args.__dict__.get('lm_prune_less_than_val', 3.17),
            top_am_threshold = args.__dict__.get('lm_top_am_threshold', -6),
        )
    beams = args.__dict__.get('lm_eval_beams', 20)

    eval_fn = dynamic_eval if not args.awmc else AWMC

    for repeat in range(args.repeats):

        all_texts, all_golds = [],[]
        wers = []
        elapsed_times = []

        for rec in tqdm(range(len(data)), total=len(data)):
            print(f'Processing {rec+1}/{len(data)}')
            
            print('\n-------\n'+data[rec]['id']+'\n-------\n')
        
            audio_spec, gold_text = data[rec]['process_fn'](data[rec])
            
            stime = time.time()
            logits = eval_fn(
                args, 
                model, 
                audio_spec, 
                args.seq_len, 
                args.overlap, 
                tokenizer,
                beam_search_fn = beamsearch
            )
            etime = time.time()
            elapsed = etime - stime
            elapsed_times.append(elapsed)  
            
            ds_factor = audio_spec.shape[-1] / logits.shape[0]
            if beamsearch is None:
                decoded, bo = decode_beams_lm([logits], decoder, beam_width=1, ds_factor=ds_factor)
                out_text = decoded[0]['text']
            else:
                run_beam_search = beamsearch(log_probs = logits, beam_width = beams)
                run_beam_search.run_search(use_tqdm = True)
                out_text = run_beam_search.return_text(idx = 0)
 

            out = normalize(out_text).lower()
            
            print(gold_text, '\n', out, '\n\n')
            
            all_texts.append(out)
            all_golds.append(gold_text)
            

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
                'repeat': f'{repeat+1}/{args.repeats}'
            }
            save_path = args.save_path
            if save_path.endswith('.pkl'): save_path = save_path.replace('.pkl', f'_{repeat+1}.pkl')
            else: save_path = save_path + f'_{repeat+1}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)

        wers.append(wer)

    print(f'Average WER: {sum(wers)/len(wers)}')
    return sum(wers)/len(wers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='earnings22', choices=datasets_functions.keys())
    parser.add_argument('--repeats', '-r', type=int, default=1, help='Number of times to repeat the evaluation')
    parser.add_argument('--save_path', '-s', type=str, default='', help='path to save')
    parser.add_argument('-nsti_s', '--nsti_seq_len', type=int, default=-1, help='Sequence length for NSTI (-1 for full recording)')
    parser.add_argument('-nsti_o', '--nsti_overlap', type=int, default=-1, help='Overlap for NSTI')

    args = lib.apply_args(parser)
    main(args)
    

#python run.py -d earnings22 -r 3 -dfa -epochs 5 -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=34 spec_augment_min_p=0.1879883950862319 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6

#CUDA_VISIBLE_DEVICES="1" python run.py -dfa -epochs 5 -seq 16384 -o 14336 -split test --dataset earnings22 -r 3 -s "./results/earnings22.json" -kwargs optim_lr=9e-5 spec_augment_freq_mask_param=34 spec_augment_min_p=0.18 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 

