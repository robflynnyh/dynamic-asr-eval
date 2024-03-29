import torch, argparse, lcasr, os, re, json
from tqdm import tqdm
from typing import Tuple
from lcasr.utils.audio_tools import processing_chain
from lcasr.eval.utils import fetch_logits, decode_beams_lm
from lcasr.utils.general import load_model
from pyctcdecode import build_ctcdecoder
from lcasr.eval.wer import word_error_rate_detail 
from whisper.normalizers import EnglishTextNormalizer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))) # for importing from parent dir
import lib
from lib import dynamic_eval
normalize = EnglishTextNormalizer()

DATA_PATH = lib.paths.datasets.rev16.test
TEST_IDS = os.path.join(DATA_PATH, 'test.txt')

def open_txt(path:str):
    with open(path, 'r') as f:
        return f.read().strip()

def fetch_data(data_path:str = DATA_PATH, ids:str = TEST_IDS):
    with open(ids, 'r') as f:
        IDS = f.read().strip().split(" ")
        IDS = [el.strip() for el in IDS if el.strip() != '']

    audio_files = [{
        'id': el,
        'path': os.path.join(data_path, "audio", el+".mp3"),
    } for el in IDS]

    text_files = [{
        'id': el,
        'text': open_txt(os.path.join(data_path, "transcripts", el+".txt"))
    } for el in IDS]

    return audio_files, text_files


def preprocess_transcript(text:str):
    return normalize(text).lower()

def process_text_and_audio_fn(rec_dict): return processing_chain(rec_dict['audio']), preprocess_transcript(rec_dict['text'])

def get_text_and_audio(split):
    assert split in ['test'], 'Split must be test'
    audio_files, text_files = fetch_data(data_path = DATA_PATH, ids = TEST_IDS)
    return_data = []
    for rec in range(len(audio_files)):
        return_data.append({
            'text': text_files[rec]['text'], 
            'audio': audio_files[rec]['path'], 
            "process_fn": process_text_and_audio_fn,
            "id": text_files[rec]['id'],
        })
    return return_data


def main(args):
    assert args.split in ['test'], 'Split must be test'
    IDS = TEST_IDS
    
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

    audio_files, text_files = fetch_data(data_path = DATA_PATH, ids = IDS)
    meetings_keys = [el['id'] for el in audio_files]
    
    all_texts = []
    all_golds = []

    beamsearch = None
    if args.beamsearch:
        beamsearch = lib.load_beamsearch(path = lib.paths.checkpoints.lm)

    for rec in tqdm(range(len(meetings_keys)), total=len(audio_files)):
        print(f'Processing {rec+1}/{len(audio_files)}')
        cur_meetings = meetings_keys[rec]
        cur_audio = audio_files[rec]['path']
        
        
        cur_text = preprocess_transcript(text_files[rec]['text'])
        assert cur_meetings == text_files[rec]['id'] and audio_files[rec]['id'] == text_files[rec]['id'], \
            f'Meeting names do not match: {cur_meetings}, {text_files[rec]["id"]}, {audio_files[rec]["id"]}'

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
    
