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

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))) # for importing from parent dir
import lib
from lib import dynamic_eval

TEST_PATH = lib.paths.datasets.earnings.test
DEV_PATH = lib.paths.datasets.earnings.dev
ALL_TEXT_PATH = lib.paths.datasets.earnings.text



def fetch_data(audio_path:str = TEST_PATH, txt_path:str = ALL_TEXT_PATH):
    with open(txt_path, 'r') as f:
        all_text_json = json.load(f)

    audio_files = [{
        'meeting': el.replace('.mp3', ''),
        'path': os.path.join(audio_path, el)
        } for el in os.listdir(audio_path) if el.endswith('.mp3')]

    text_files = [{
        'meeting': el['meeting'],
        'text': all_text_json[el['meeting']]
        } for el in audio_files]
 

    return audio_files, text_files



def preprocess_transcript(text:str):
    text = text.lower()
    text = text.replace('<silence>', '')
    text = text.replace('<inaudible>', '')
    text = text.replace('<laugh>', '')
    text = text.replace('<noise>', '')
    text = text.replace('<affirmative>', '')
    text = text.replace('<crosstalk>', '')    
    text = text.replace('…', '')
    text = text.replace(',', '')
    text = text.replace('-', ' ')
    text = text.replace('.', '')
    text = text.replace('?', '')
    text = re.sub(' +', ' ', text)
    return normalize(text).lower()

def process_text_and_audio_fn(rec_dict): return processing_chain(rec_dict['audio']), preprocess_transcript(rec_dict['text'])

def get_text_and_audio(split):
    assert split in ['test', 'dev'], f'Split must be either test or dev (got {args.split})'
    data_path = TEST_PATH if split == 'test' else DEV_PATH
    audio_files, text_files = fetch_data(audio_path=data_path, txt_path=ALL_TEXT_PATH)
    return_data = []
    for rec in range(len(audio_files)):
        return_data.append({
            'id': audio_files[rec]['meeting'],
            'text': text_files[rec]['text'], 
            'audio': audio_files[rec]['path'], 
            "process_fn": process_text_and_audio_fn
        })
    return return_data

def main(args):
    assert args.split in ['test', 'dev'], f'Split must be either test or dev (got {args.split})'
    data_path = TEST_PATH if args.split == 'test' else DEV_PATH
    
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

    audio_files, text_files = fetch_data(audio_path=data_path, txt_path=ALL_TEXT_PATH)
    meetings_keys = [el['meeting'] for el in audio_files]
    
    beamsearch = None
    if args.beamsearch: beamsearch = lib.load_beamsearch(path = lib.paths.checkpoints.lm)

    all_texts, all_golds = [],[]

    for rec in tqdm(range(len(meetings_keys)), total=len(audio_files)):

        print(f'Processing {rec+1}/{len(audio_files)}')
        cur_meetings = meetings_keys[rec]
        cur_audio = audio_files[rec]['path']
        
        
        cur_text = preprocess_transcript(text_files[rec]['text'])
        assert cur_meetings == text_files[rec]['meeting'] and audio_files[rec]['meeting'] == text_files[rec]['meeting'], \
            f'Meeting names do not match: {cur_meetings}, {text_files[rec]["meeting"]}, {audio_files[rec]["meeting"]}'

        audio_spec = processing_chain(cur_audio)
        print('\n-------\n'+cur_meetings+'\n-------\n')
        
        logits = dynamic_eval(
            args, 
            model, 
            audio_spec, 
            args.seq_len, 
            args.overlap, 
            tokenizer,
            beam_search_fn = beamsearch
        )

        ds_factor = audio_spec.shape[-1] / logits.shape[0]
        decoded, bo = decode_beams_lm([logits], decoder, beam_width=1, ds_factor=ds_factor)
        out = normalize(decoded[0]['text']).lower()
        
        print(cur_text, '\n', out, '\n\n')
        
        all_texts.append(out)
        all_golds.append(cur_text)
        
    
    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    print(f'WER: {wer}')

    if args.log != '':
        with open(args.log, 'a') as f:
            f.write(f'{args.checkpoint}\t overlap: {args.overlap}\t seq_len: {args.seq_len}\t WER: {wer}\n')

    return wer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = lib.apply_args(parser)
    
    main(args)
    
