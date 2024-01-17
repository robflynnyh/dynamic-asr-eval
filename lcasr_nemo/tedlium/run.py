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
import torchaudio
import time

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))) # for importing from parent dir
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

def fetch_utterances(stm_path:str, waveform:torch.Tensor, sr:int = 16000):
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
        #print(int(float(start) * sr), int(float(end) * sr))
        text = re.sub(r" '([a-z])", r"'\1", text)
        text = re.sub(r" +", r" ", text)
        utterances.append({
            'start': float(start), 
            'end': float(end), 
            'text': text, 
            'start_frame': int(float(start) * sr),
            'end_frame': int(float(end) * sr),
            'waveform': waveform[:, int(float(start) * sr):int(float(end) * sr)]
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

    

    tokenizer = lib.load_tokenizer(path = args.tokenizer_path)

    model = lib.load_model(path = args.checkpoint)
 
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()

    #vocab = [tokenizer.ids_to_text(id) for id in range(tokenizer.vocab_size)] + [""]
    decoder = build_ctcdecoder(tokenizer.vocab, kenlm_model_path=None, alpha=None, beta=None)


    audio_files, text_files = fetch_data(path=data_path)
    paired = dict(zip(audio_files, text_files))
    
    all_texts = []
    all_golds = []

    beamsearch = None
    if args.beamsearch:
        beamsearch = lib.load_beamsearch(path = lib.paths.checkpoints.lm)

    for rec in tqdm(range(len(audio_files)), total=len(audio_files)):
        print(f'Processing {rec+1}/{len(audio_files)}') if args.verbose else None   

        audio_wav,_= torchaudio.load(audio_files[rec])
        print('\n\n'+paired[audio_files[rec]]+'\n\n') if args.verbose else None
        stm_path = paired[audio_files[rec]]
        gold_text, timings, remove_timings = proc_stm_and_timings(stm_path=stm_path)

        utterances, gold_text = fetch_utterances(stm_path=stm_path, waveform=audio_wav)
        
        stime = time.time()
        utterances_out = dynamic_eval(
            args,
            model,
            utterances,
            args.seq_len, 
            args.overlap, 
            tokenizer,
        )
        
        etime = time.time()
        print(f'Inference time: {etime-stime}')
      
        for i, utt in enumerate(utterances_out):
            decoded, bo = decode_beams_lm([utt['probs']], decoder, beam_width=1, ds_factor=None)
            print(bo.text)
            utterances_out[i]['text'] = bo.text

      

        text = ' '.join([el['text'].strip() for el in utterances_out])
        out = normalize(text).lower()

        gold_text = normalize(gold_text).lower()    
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
    
