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
    if args.dataset2 == 'rev16': assert args.split == 'test', 'Split must be test for rev16'

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

    data_a = datasets_functions[args.dataset](args.split)
    data_b = datasets_functions[args.dataset2](args.split)

    print(f'Dataset A ({args.dataset}): {len(data_a)} records')
    print(f'Dataset B ({args.dataset2}): {len(data_b)} records')

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

    args_dict = vars(args).copy()
    args_dict['epochs'] = 0
    baseline_args = argparse.Namespace(**args_dict)

    for repeat in range(args.repeats):
        print(f'\n=== Repeat {repeat+1}/{args.repeats} ===')

        a_baseline = None
        b_baseline = None
        a_to_b = []
        a_to_a_loo = []

        print(f'Baseline on A ({args.dataset})')
        golds, preds = [], []
        for idx in tqdm(range(len(data_a)), total=len(data_a)):
            audio_spec, gold_text = data_a[idx]['process_fn'](data_a[idx])
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
        a_baseline = {"wer": wer, "words": words, "ins_rate": ins_rate, "del_rate": del_rate, "sub_rate": sub_rate}
        print(f'A baseline WER: {wer}')

        print(f'Baseline on B ({args.dataset2})')
        golds, preds = [], []
        for idx in tqdm(range(len(data_b)), total=len(data_b)):
            audio_spec, gold_text = data_b[idx]['process_fn'](data_b[idx])
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
        b_baseline = {"wer": wer, "words": words, "ins_rate": ins_rate, "del_rate": del_rate, "sub_rate": sub_rate}
        print(f'B baseline WER: {wer}')

        print('A-X (adapt on A[i], eval on B and A leave-one-out)')
        for i in tqdm(range(len(data_a)), total=len(data_a)):
            audio_spec, gold_text = data_a[i]['process_fn'](data_a[i])
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

            golds, preds = [], []
            for j in range(len(data_b)):
                audio_spec, gold_text = data_b[j]['process_fn'](data_b[j])
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
            a_to_b.append({"wer": wer, "words": words, "ins_rate": ins_rate, "del_rate": del_rate, "sub_rate": sub_rate})

            other_a_idxs = [k for k in range(len(data_a)) if k != i]
            golds, preds = [], []
            for k in other_a_idxs:
                audio_spec, gold_text = data_a[k]['process_fn'](data_a[k])
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
            a_to_a_loo.append({"wer": wer, "words": words, "ins_rate": ins_rate, "del_rate": del_rate, "sub_rate": sub_rate})

            for p, u in zip(model.parameters(), original_model_params):
                p.data = u.data.to(p.device)

        results = {
            'a_baseline': a_baseline,
            'b_baseline': b_baseline,
            'a_to_b': a_to_b,
            'a_to_a_loo': a_to_a_loo,
            'dataset_a': args.dataset,
            'dataset_b': args.dataset2,
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
    parser.add_argument('--dataset', '-d', type=str, default='earnings22', choices=datasets_functions.keys(), required=True)
    parser.add_argument('--dataset2', '-d2',  type=str, default='', choices=datasets_functions.keys(), required=True)
    parser.add_argument('--repeats', '-r', type=int, default=1, help='Number of times to repeat the evaluation')
    parser.add_argument('--save_path', '-s', type=str, default='', help='path to save')
    parser.add_argument('--adapt_overlap', '-ao', type=int, default=None, help='Overlap used during adaptation passes only. If unset, adaptation uses --overlap (current behavior).')

    args = lib.apply_args(parser)
    main(args)


#python run.py -d earnings22 -r 3 -dfa -epochs 5 -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=34 spec_augment_min_p=0.1879883950862319 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6

#CUDA_VISIBLE_DEVICES="1" python run.py -dfa -epochs 5 -seq 16384 -o 14336 -split test --dataset earnings22 -r 3 -s "./results/earnings22.json" -kwargs optim_lr=9e-5 spec_augment_freq_mask_param=34 spec_augment_min_p=0.18 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0
