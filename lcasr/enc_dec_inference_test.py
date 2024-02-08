import torch, argparse, lcasr, os, re, json
from tqdm import tqdm
from typing import Tuple
from lcasr.utils.audio_tools import processing_chain
from lcasr.eval.utils import fetch_logits, decode_beams_lm
from lcasr.utils.general import load_model, get_model_class
from pyctcdecode import build_ctcdecoder
from lcasr.eval.wer import word_error_rate_detail 
#from lcasr.eval.dynamic_eval import dynamic_eval
from whisper.normalizers import EnglishTextNormalizer

normalize = EnglishTextNormalizer()

import sys
import os.path

import lib
from lib import enc_dec_inference

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
    model = load_model(args.config, model_class=get_model_class(args.config), vocab_size=len(tokenizer))
    tparams = model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()

    data = datasets_functions[args.dataset](args.split)

    all_texts, all_golds = [],[]
  
    for rec in tqdm(range(len(data)), total=len(data)):
        print(f'Processing {rec+1}/{len(data)}')
        
        print('\n-------\n'+data[rec]['id']+'\n-------\n')
    
        audio_spec, gold_text = data[rec]['process_fn'](data[rec])
        
        model_out = enc_dec_inference(
            model = model,
            spec = audio_spec,
            seq_len = args.seq_len,
            overlap = 0,
            tokenizer = tokenizer,
            use_tqdm = True
        )

        out = normalize(model_out).lower()
        
        print(gold_text, '\n', out, '\n\n')
        
        all_texts.append(out)
        all_golds.append(gold_text)
        
            

    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    print(f'WER: {wer}')

    if args.log != '':
        with open(args.log, 'a') as f:
            f.write(f'{args.checkpoint}\t overlap: {args.overlap}\t seq_len: {args.seq_len}\t WER: {wer}\n')

    return wer
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='earnings22', choices=datasets_functions.keys())
    args = lib.apply_args(parser)
    main(args)
    

#python run.py -d earnings22 -r 3 -dfa -epochs 5 -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=34 spec_augment_min_p=0.1879883950862319 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6

