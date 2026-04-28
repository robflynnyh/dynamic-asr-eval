import torch, argparse, lcasr, os, re, json
from tqdm import tqdm
from typing import Tuple
from lcasr.utils.audio_tools import processing_chain
from lcasr.utils.general import load_model
from lcasr.decoding.greedy import GreedyCTCDecoder
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

from tedlium.run import get_text_and_audio as get_text_and_audio_tedlium

DEFAULT_SPEAKER_MANIFEST = os.path.join(
    os.path.dirname(__file__),
    "results",
    "gender_eval_tedlium",
    "speaker_manifest_15x15.json",
)

datasets_functions = {'tedlium': get_text_and_audio_tedlium}


def load_speaker_manifest(path: str):
    with open(path, 'r') as f:
        manifest = json.load(f)
    speaker_gender = {}
    for row in manifest['female']:
        speaker_gender[row['talk_id'] + '.sph'] = 'F'
    for row in manifest['male']:
        speaker_gender[row['talk_id'] + '.sph'] = 'M'
    return manifest, speaker_gender


def main(args):
    assert args.split in ['test', 'dev', 'train'], f'Split must be either test, dev, or train (got {args.split})'
    if args.dataset == 'rev16': assert args.split == 'test', 'Split must be test for rev16'

    speaker_manifest, speaker_gender = load_speaker_manifest(args.speaker_manifest)
    print(f"Loaded speaker manifest from {args.speaker_manifest}: {len(speaker_manifest['female'])} female, {len(speaker_manifest['male'])} male")
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
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

    decoder = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=model.decoder.num_classes-1)

    test = datasets_functions["tedlium"]("test")
    dev = datasets_functions["tedlium"]("dev")
    train = datasets_functions["tedlium"]("train")
   
    all_data = test + dev + train

    eval_data = [rec for rec in all_data if os.path.basename(rec['id']) in speaker_gender]
    males = [rec for rec in eval_data if speaker_gender[os.path.basename(rec['id'])] == "M"]
    females = [rec for rec in eval_data if speaker_gender[os.path.basename(rec['id'])] == "F"]
    print(f'Female data: {[os.path.basename(el["id"]) for el in females]}')
    print(f'Male data: {[os.path.basename(el["id"]) for el in males]}')
    assert len(females) + len(males) == len(eval_data), "Data filtered incorrectly"

    print(f'Total data: {len(eval_data)}')

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

    adapt_overlap = args.adapt_overlap if args.adapt_overlap is not None else args.overlap
    if adapt_overlap != args.overlap:
        print(f'Using adapt_overlap={adapt_overlap} for adaptation (eval overlap={args.overlap})')

    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach().cpu() for p in original_model_params]

    def transcribe_from_logits(logits, audio_spec):
        if beamsearch is None:
            out_text = decoder(torch.as_tensor(logits))
        else:
            run_beam_search = beamsearch(log_probs = logits, beam_width = beams)
            run_beam_search.run_search(use_tqdm = True)
            out_text = run_beam_search.return_text(idx = 0)
        out = normalize(out_text).lower()
        return out

    # clone
    args_dict = vars(args).copy()
    args_dict['epochs'] = 0
    baseline_args = argparse.Namespace(**args_dict)

    for repeat in range(args.repeats):
        print(f'\n=== Repeat {repeat+1}/{args.repeats} ===')

        male_baseline = None
        female_baseline = None
        male_to_male = []
        female_to_female = []
        male_to_female = []
        female_to_male = []

        male_idxs = [i for i in range(len(males))]
        golds, preds = [], []
        print('Male baseline')
        for male_idx in tqdm(male_idxs, total=len(male_idxs)):
            audio_spec, gold_text = males[male_idx]['process_fn'](males[male_idx])
            logits = eval_fn(
                baseline_args,
                model,
                audio_spec,
                args.seq_len,
                args.overlap,
                tokenizer,
                beam_search_fn = beamsearch
            )
            pred = transcribe_from_logits(logits, audio_spec)
            golds.append(gold_text)
            preds.append(pred)
        wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=preds, references=golds)
        male_baseline = {"wer": wer, "words": words, "ins_rate": ins_rate, "del_rate": del_rate, "sub_rate": sub_rate}
        print(f'Male baseline WER: {wer}')

        print('Female baseline')
        female_idxs = [i for i in range(len(females))]
        golds, preds = [], []
        for female_idx in tqdm(female_idxs, total=len(female_idxs)):
            audio_spec, gold_text = females[female_idx]['process_fn'](females[female_idx])
            logits = eval_fn(
                baseline_args,
                model,
                audio_spec,
                args.seq_len,
                args.overlap,
                tokenizer,
                beam_search_fn = beamsearch
            )
            pred = transcribe_from_logits(logits, audio_spec)
            golds.append(gold_text)
            preds.append(pred)
        wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=preds, references=golds)
        female_baseline = {"wer": wer, "words": words, "ins_rate": ins_rate, "del_rate": del_rate, "sub_rate": sub_rate}
        print(f'Female baseline WER: {wer}')

        print("Male-X")
        male_idxs = [i for i in range(len(males))]
        for male_idx in tqdm(male_idxs, total=len(male_idxs)):
            audio_spec, gold_text = males[male_idx]['process_fn'](males[male_idx])
            print(args.epochs)
            _, updated_parameters = eval_fn(
                args,
                model,
                audio_spec,
                args.seq_len,
                adapt_overlap,
                tokenizer,
                beam_search_fn = beamsearch,
                return_params = True
            )
            for p, u in zip(model.parameters(), updated_parameters):
                p.data = u.data.to(p.device)

            other_male_idxs = [i for i in range(len(males)) if i != male_idx]
            golds, preds = [], []
            for other_male_idx in other_male_idxs:
                audio_spec, gold_text = males[other_male_idx]['process_fn'](males[other_male_idx])
                logits = eval_fn(
                    baseline_args,
                    model,
                    audio_spec,
                    args.seq_len,
                    args.overlap,
                    tokenizer,
                    beam_search_fn = beamsearch
                )
                pred = transcribe_from_logits(logits, audio_spec)
                golds.append(gold_text)
                preds.append(pred)
            wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=preds, references=golds)
            male_to_male.append({"wer": wer, "words": words, "ins_rate": ins_rate, "del_rate": del_rate, "sub_rate": sub_rate})

            female_idxs = [i for i in range(len(females))]
            golds, preds = [], []
            for female_idx in female_idxs:
                audio_spec, gold_text = females[female_idx]['process_fn'](females[female_idx])
                logits = eval_fn(
                    baseline_args,
                    model,
                    audio_spec,
                    args.seq_len,
                    args.overlap,
                    tokenizer,
                    beam_search_fn = beamsearch
                )
                pred = transcribe_from_logits(logits, audio_spec)
                golds.append(gold_text)
                preds.append(pred)
            wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=preds, references=golds)
            male_to_female.append({"wer": wer, "words": words, "ins_rate": ins_rate, "del_rate": del_rate, "sub_rate": sub_rate})

            for p, u in zip(model.parameters(), original_model_params):
                p.data = u.data.to(p.device)

        print("Female-X")
        female_idxs = [i for i in range(len(females))]
        for female_idx in tqdm(female_idxs, total=len(female_idxs)):
            audio_spec, gold_text = females[female_idx]['process_fn'](females[female_idx])
            _, updated_parameters = eval_fn(
                args,
                model,
                audio_spec,
                args.seq_len,
                adapt_overlap,
                tokenizer,
                beam_search_fn = beamsearch,
                return_params = True
            )
            for p, u in zip(model.parameters(), updated_parameters):
                p.data = u.data.to(p.device)

            other_female_idxs = [i for i in range(len(females)) if i != female_idx]
            golds, preds = [], []
            for other_female_idx in other_female_idxs:
                audio_spec, gold_text = females[other_female_idx]['process_fn'](females[other_female_idx])
                logits = eval_fn(
                    baseline_args,
                    model,
                    audio_spec,
                    args.seq_len,
                    args.overlap,
                    tokenizer,
                    beam_search_fn = beamsearch
                )
                pred = transcribe_from_logits(logits, audio_spec)
                golds.append(gold_text)
                preds.append(pred)
            wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=preds, references=golds)
            female_to_female.append({"wer": wer, "words": words, "ins_rate": ins_rate, "del_rate": del_rate, "sub_rate": sub_rate})

            male_idxs = [i for i in range(len(males))]
            golds, preds = [], []
            for male_idx in male_idxs:
                audio_spec, gold_text = males[male_idx]['process_fn'](males[male_idx])
                logits = eval_fn(
                    baseline_args,
                    model,
                    audio_spec,
                    args.seq_len,
                    args.overlap,
                    tokenizer,
                    beam_search_fn = beamsearch
                )
                pred = transcribe_from_logits(logits, audio_spec)
                golds.append(gold_text)
                preds.append(pred)
            wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=preds, references=golds)
            female_to_male.append({"wer": wer, "words": words, "ins_rate": ins_rate, "del_rate": del_rate, "sub_rate": sub_rate})

            for p, u in zip(model.parameters(), original_model_params):
                p.data = u.data.to(p.device)

        results = {
            'male_baseline': male_baseline,
            'female_baseline': female_baseline,
            'male_to_male': male_to_male,
            'male_to_female': male_to_female,
            'female_to_female': female_to_female,
            'female_to_male': female_to_male,
            'args_dict': vars(args),
            'repeat': f'{repeat+1}/{args.repeats}'
        }

        if args.save_path != '':
            save_path = args.save_path
            if save_path.endswith('.pkl'): save_path = save_path.replace('.pkl', f'_{repeat+1}.pkl')
            else: save_path = save_path + f'_{repeat+1}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)

            print(f'Finished and saved to {save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', '-r', type=int, default=1, help='Number of times to repeat the evaluation')
    parser.add_argument('--save_path', '-s', type=str, default='', help='path to save')
    parser.add_argument('--adapt_overlap', '-ao', type=int, default=None, help='Overlap used during adaptation passes only. If unset, adaptation uses --overlap (current behavior).')
    parser.add_argument('--speaker_manifest', type=str, default=DEFAULT_SPEAKER_MANIFEST, help='Path to the gender selection manifest')

    args = lib.apply_args(parser)
    args.dataset = 'tedlium'
    
    main(args)
    

#python run.py -d earnings22 -r 3 -dfa -epochs 5 -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=34 spec_augment_min_p=0.1879883950862319 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6

#CUDA_VISIBLE_DEVICES="1" python run.py -dfa -epochs 5 -seq 16384 -o 14336 -split test --dataset earnings22 -r 3 -s "./results/earnings22.json" -kwargs optim_lr=9e-5 spec_augment_freq_mask_param=34 spec_augment_min_p=0.18 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 

