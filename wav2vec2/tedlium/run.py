import torch, argparse, lcasr, os, re, json
from tqdm import tqdm
from typing import Tuple
from lcasr.eval.utils import fetch_logits, decode_beams_lm

from lcasr.eval.wer import word_error_rate_detail 
from whisper.normalizers import EnglishTextNormalizer
from lcasr.decoding.greedy import GreedyCTCDecoder
normalize = EnglishTextNormalizer()
from typing import List, Dict
from lcasr.utils.audio_tools import processing_chain, total_frames
from lcasr.utils.general import load_model
from lcasr.eval.utils import zero_out_spectogram, decode_beams_lm, fetch_logits

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))) # for importing from parent dir
import lib

TEST_PATH = lib.paths.datasets.tedlium.test
DEV_PATH = lib.paths.datasets.tedlium.dev



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




def main(args):
    assert args.split in ['test', 'dev'], f'Split must be either test or dev (got {args.split})'
    data_path = TEST_PATH if args.split == 'test' else DEV_PATH

    
    model, processor = lib.load_pretrained_model(args.checkpoint)
    print(f'Loaded model from {args.checkpoint}')
    # print total number of parameters (M)
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()
    tokenizer = processor.tokenizer
    vocab = [el[0] for el in tokenizer.vocab.items()] 
    space_id = vocab.index("|")
    vocab[space_id] = " "
    tokenizer.blank_id = 0
    tokenizer.vocab_list = vocab
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = tokenizer.blank_id)
    print(tokenizer.vocab)
    audio_files, text_files = fetch_data(path=data_path)
    paired = dict(zip(audio_files, text_files))
    
    all_texts = []
    all_golds = []

    for rec in tqdm(range(len(audio_files)), total=len(audio_files)):
        print(f'Processing {rec+1}/{len(audio_files)}') if args.verbose else None   

    
        print('\n\n'+paired[audio_files[rec]]+'\n\n') if args.verbose else None
        stm_path = paired[audio_files[rec]]
        gold_text, timings, remove_timings = proc_stm_and_timings(stm_path=stm_path)

        audio_spec = lib.preprocess(audio_files[rec])
        #audio_spec = zero_out_spectogram(spec = audio_spec, remove_timings = remove_timings)
        
 
        logits = lib.dynamic_eval(args, model, audio_spec, args.seq_len, args.overlap, tokenizer, processor)
 
        ds_factor = audio_spec.shape[-1] / logits.shape[0]

        ds_factor = audio_spec.shape[-1] / logits.shape[0]
        text = decoder(torch.as_tensor(logits)).lower()
        out = normalize(text).lower()
        
        print(gold_text) if args.verbose else None
        print(out) if args.verbose else None
        all_texts.append(out)
        all_golds.append(gold_text)
        break

        
    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    print(f'WER: {wer}')

    if args.log != '':
        with open(args.log, 'a') as f:
            f.write(f'{args.checkpoint}\t overlap: {args.overlap}\t seq_len: {args.seq_len}\t WER: {wer}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = lib.apply_args(parser)
    main(args)
    
