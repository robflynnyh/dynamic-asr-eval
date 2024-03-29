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
from lib import dynamic_eval, AWMC, prepare_chunks
import ffmpeg

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

def get_audio_length(fname:str) -> float:
    dur = ffmpeg.probe(fname)['format']['duration']
    return float(dur)

def main(args):
    assert args.split in ['test', 'dev'], f'Split must be either test or dev (got {args.split})'
    assert args.dataset != 'chime6', 'will throw an error when checking length of each audio file on chime6 (i.e not implemented for this dataset, the eval in the paper looks at earnings22)'
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
    overlap = args.nsti_overlap

    data_test = datasets_functions[args.dataset]("test")
    data_dev = datasets_functions[args.dataset]("dev")
    data_all = data_test + data_dev
    data = [el for el in data_all if get_audio_length(el['audio'])/60 >= 60.0]

    print([get_audio_length(data[i]['audio']) / 60 for i in range(len(data))])
    
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
        
        
        for rec in tqdm(range(len(data)), total=len(data)):
          
            print(f'Processing {rec+1}/{len(data)}')
            
            print('\n-------\n'+data[rec]['id']+'\n-------\n')
        
            audio_spec, gold_text = data[rec]['process_fn'](data[rec])
            
            spec_n = audio_spec.shape[-1]
            seq_len = args.nsti_seq_len if args.nsti_seq_len != -1 else spec_n
            
            if args.epochs == 0: # just eval over whole sequence if not performing NSTI
                seq_len = spec_n
                overlap = 0 

            model_outputs = {}
            all_logits, logit_count = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1)), torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1))
            
            training_data, training_keys = prepare_chunks(audio_spec, seq_len, overlap)
            training_keys = list(training_data.keys())
    
            pbar = tqdm(training_keys, total=len(training_keys))
            for i in pbar:
                cur_spec = training_data[i]
                logits = eval_fn(
                    args, 
                    model, 
                    cur_spec, 
                    args.seq_len, 
                    args.overlap, 
                    tokenizer,
                    beam_search_fn = beamsearch
                )
                logits = torch.exp(torch.as_tensor(logits)[None]).detach().cpu()
              
                ds_len = logits.shape[-2]
                u_len = cur_spec.shape[-1]
                ratio = u_len / ds_len
                overlap_ds = int(overlap / ratio)
                model_outputs[i] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}

            logit_position = 0
            for i in sorted(list(model_outputs.keys())):
                logits, ds_len, overlap_ds = model_outputs[i]['logits'], model_outputs[i]['ds_len'], model_outputs[i]['overlap_ds']
                logit_position -= overlap_ds if i != 0 else 0
                logit_count[:, logit_position:logit_position+ds_len, :] += 1
                all_logits[:, logit_position:logit_position+ds_len, :] += logits
                logit_position += ds_len

            B,N,C = all_logits.shape
            all_logits = all_logits[logit_count.sum(dim=-1) != 0]
            all_logits = all_logits.reshape(B,-1,C)
            logit_count = logit_count[logit_count.sum(dim=-1) != 0]
            logit_count = logit_count.reshape(B,-1,C)
            logits = all_logits / logit_count
            logits = torch.log(logits) # convert to log 
            
            logits = logits.squeeze(0).numpy()
            
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
    parser.add_argument('-nsti_o', '--nsti_overlap', type=int, default=0, help='Overlap for NSTI')

    args = lib.apply_args(parser)
    main(args)
    

#python run.py -d earnings22 -r 3 -dfa -epochs 5 -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=34 spec_augment_min_p=0.1879883950862319 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6

#CUDA_VISIBLE_DEVICES="1" python run.py -dfa -epochs 5 -seq 16384 -o 14336 -split test --dataset earnings22 -r 3 -s "./results/earnings22.json" -kwargs optim_lr=9e-5 spec_augment_freq_mask_param=34 spec_augment_min_p=0.18 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 

