import torch, lcasr, os, re
import argparse
from tqdm import tqdm
from typing import List
from lcasr.utils.audio_tools import processing_chain, total_frames
from lcasr.utils.general import load_model
from lcasr.eval.utils import zero_out_spectogram, decode_beams_lm, fetch_logits
#from lcasr.eval.dynamic_eval import dynamic_eval
from lcasr.eval.wer import word_error_rate_detail 
from pyctcdecode import build_ctcdecoder
import time

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))) # for importing from parent dir
import lib
from lib import dynamic_eval

TEST_PATH = lib.paths.datasets.tedlium.test
DEV_PATH = lib.paths.datasets.tedlium.dev

from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()

def open_stm(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    return lines

def proc_stm_and_timings(stm_path:str):
    stm = open_stm(stm_path)
    all_text = ""
    timings = []
    remove_timings = []
    for line in stm:
        sline = line.split(' ')
        if len(sline) < 6: continue
        a_id, s_id, spk, start, end, meta = sline[:6]
        text = ' '.join(sline[6:])
        if text == 'ignore_time_segment_in_scoring':
            remove_timings.append({'start': float(start), 'end': float(end)})
            continue
        all_text += text + ' '
        timings.append({'start': float(start), 'end': float(end)})
    all_text = all_text.strip()
    # regex to do all of the above
    # i.e replace space followed by a apostrophe followed by a letter with just the apostrophe and letter
    all_text = re.sub(r" '([a-z])", r"'\1", all_text)
    # remove multiple spaces
    all_text = re.sub(r" +", r" ", all_text)
    return all_text, timings, remove_timings

def fetch_utterances(stm_path:str, spectogram:torch.Tensor):
    stm = open_stm(stm_path)
    utterances = []
    for line in stm:
        sline = line.split(' ')
        if len(sline) < 6:
            continue
        a_id, s_id, spk, start, end, meta = sline[:6]
        text = ' '.join(sline[6:])
        if text == 'ignore_time_segment_in_scoring':
            continue
        utterances.append({
            'start': float(start), 
            'end': float(end), 
            'text': text, 
            'start_frame': total_frames(float(start)), 
            'end_frame': total_frames(float(end)),
            'spectogram': spectogram[:, :, total_frames(float(start)):total_frames(float(end))]
        })
    
    all_text = " ".join([el['text'] for el in utterances])
    all_text = re.sub(r" '([a-z])", r"'\1", all_text)
    all_text = re.sub(r" +", r" ", all_text)
        
    return utterances, all_text


def fetch_data(path:str = TEST_PATH):
    audio_path = os.path.join(path, 'sph')
    audio_files = [os.path.join(audio_path, el) for el in os.listdir(audio_path) if el.endswith('.sph')]
    audio_files.sort()
    text_path = os.path.join(path, 'stm')
    text_files = [os.path.join(text_path, el) for el in os.listdir(text_path) if el.endswith('.stm')]
    text_files.sort()
    assert len(audio_files) == len(text_files), 'Number of audio files and text files must match'
    return audio_files, text_files


def process_text_and_audio_fn(rec_dict):
    audio, text = rec_dict['audio'], rec_dict['text']
    gold_text, _, remove_timings = proc_stm_and_timings(stm_path=text)
    audio_spec = processing_chain(audio)
    audio_spec = zero_out_spectogram(spec = audio_spec, remove_timings = remove_timings)
    return audio_spec, normalize(gold_text).lower()

def get_text_and_audio(split):
    assert split in ['test', 'dev', 'train'], f'Split must be either test or dev train (got {args.split})'
    data_path = TEST_PATH if split == 'test' else DEV_PATH
    if split == 'train': data_path = lib.paths.datasets.tedlium.train
    
    audio_files, text_files = fetch_data(path=data_path)
    return_data = []
    for rec in range(len(audio_files)):
        return_data.append({
            'id': audio_files[rec],
            'text': text_files[rec], 
            'audio': audio_files[rec], 
            "process_fn": process_text_and_audio_fn
        })

    return return_data



def main(args):
    assert args.split in ['test', 'dev', 'train'], f'Split must be either test or dev or train (got {args.split})'
    
    if args.split == 'test': data_path = TEST_PATH
    elif args.split == 'dev': data_path = DEV_PATH
    elif args.split == 'train': data_path = lib.paths.datasets.tedlium.train
    else: raise ValueError(f'Invalid split: {args.split}')
 
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


    audio_files, text_files = fetch_data(path=data_path)
    paired = dict(zip(audio_files, text_files))
    
    all_texts = []
    all_golds = []

    beamsearch = None
    if args.beamsearch:
        beamsearch = lib.load_beamsearch(path = lib.paths.checkpoints.lm)

    for rec in tqdm(range(len(audio_files)), total=len(audio_files)):
        print(f'Processing {rec+1}/{len(audio_files)}') if args.verbose else None   

        audio_spec = processing_chain(audio_files[rec])
        print('\n\n'+paired[audio_files[rec]]+'\n\n') if args.verbose else None
        stm_path = paired[audio_files[rec]]
        gold_text, timings, remove_timings = proc_stm_and_timings(stm_path=stm_path)

        audio_spec = zero_out_spectogram(spec = audio_spec, remove_timings = remove_timings)
        
        stime = time.time()
        logits = dynamic_eval(
            args, 
            model, 
            audio_spec, 
            args.seq_len, 
            args.overlap, 
            tokenizer,
            beam_search_fn = beamsearch
        )

        etime = time.time()
        print(f'Inference time: {etime-stime}')
        ds_factor = audio_spec.shape[-1] / logits.shape[0]
        decoded, bo = decode_beams_lm([logits], decoder, beam_width=1, ds_factor=ds_factor)

        all_text = normalize(decoded[0]['text']).lower()
        all_text = all_text[:-1].strip() if all_text.endswith('.') else all_text.strip()
        gold_text = normalize(gold_text).lower()    
        print(gold_text) if args.verbose else None
        print(all_text) if args.verbose else None
        all_texts.append(all_text)
        all_golds.append(gold_text)
        #break

        
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
    
