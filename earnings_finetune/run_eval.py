from omegaconf import OmegaConf
import os
import json

paths_dir = os.path.join(os.path.dirname(__file__), './paths.yaml')
assert os.path.exists(paths_dir), 'paths.yaml not found'

paths = OmegaConf.load(paths_dir)
TEST_PATH = paths.earnings22.test
DEV_PATH = paths.earnings22.dev
ALL_TEXT_PATH = paths.earnings22.text
from whisper.normalizers import EnglishTextNormalizer
from lcasr.eval.wer import word_error_rate_detail
from lcasr.eval.utils import fetch_logits, decode_beams_lm
from lcasr.utils.audio_tools import processing_chain
import re
from pyctcdecode import build_ctcdecoder
normalize = EnglishTextNormalizer()

from tqdm import tqdm

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


class EvalRunner:
    def __init__(self, tokenizer, split = 'dev'):
        self.tokenizer = tokenizer
        self.split = split
        assert split in ['test', 'dev'], 'Split must be either test or dev'
        data_path = TEST_PATH if split == 'test' else DEV_PATH
        self.vocab = [tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())] + [""]
        self.decoder = build_ctcdecoder(self.vocab, kenlm_model_path=None, alpha=None, beta=None)
        self.audio_files, self.text_files = fetch_data(audio_path=data_path, txt_path=ALL_TEXT_PATH)
        self.meetings_keys = [el['meeting'] for el in self.audio_files]
        
    def run_eval(self, model, device, seq_len, overlap):
        all_texts, all_golds = [], []
        model.device = device
        model.eval()

        for rec in tqdm(range(len(self.audio_files)), total=len(self.audio_files)):
            print(f'Processing {rec+1}/{len(self.audio_files)}')
            cur_meetings = self.meetings_keys[rec]
            cur_audio = self.audio_files[rec]['path']

            cur_text = preprocess_transcript(self.text_files[rec]['text'])
            assert cur_meetings == self.text_files[rec]['meeting'] and self.audio_files[rec]['meeting'] == self.text_files[rec]['meeting'], \
                f'Meeting names do not match: {cur_meetings}, {self.text_files[rec]["meeting"]}, {self.audio_files[rec]["meeting"]}'  
            
            audio_spec = processing_chain(cur_audio)
            print('\n-------\n'+cur_meetings+'\n-------\n')

            logits = fetch_logits(None, model, audio_spec, seq_len, overlap, self.tokenizer)

            ds_factor = audio_spec.shape[-1] / logits.shape[0]
            decoded, bo = decode_beams_lm([logits], self.decoder, beam_width=1, ds_factor=ds_factor)
            out = normalize(decoded[0]['text']).lower()
            
            print(cur_text, '\n', out, '\n\n')
            
            all_texts.append(out)
            all_golds.append(cur_text)

        wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)
        print(f'WER: {wer}')
        model.train()
        return wer




