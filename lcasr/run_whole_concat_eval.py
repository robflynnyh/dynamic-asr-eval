import argparse
import copy
import pickle

import lcasr
import torch
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

import lib
from earnings22.run import get_text_and_audio as get_text_and_audio_earnings22
from chime6.run import get_text_and_audio as get_text_and_audio_chime6
from tedlium.run import get_text_and_audio as get_text_and_audio_tedlium
from rev16.run import get_text_and_audio as get_text_and_audio_rev16
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.eval.wer import word_error_rate_detail
from lcasr.utils.general import load_model
from lib import AWMC, dynamic_eval
from run_half_concat_eval import adapt_on_concat_only, concatenate_specs

normalize = EnglishTextNormalizer()


datasets_functions = {
    'earnings22': get_text_and_audio_earnings22,
    'chime6': get_text_and_audio_chime6,
    'tedlium': get_text_and_audio_tedlium,
    'rev16': get_text_and_audio_rev16,
}


def score_texts(preds, refs):
    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=preds, references=refs)
    return {
        'wer': wer,
        'words': words,
        'ins_rate': ins_rate,
        'del_rate': del_rate,
        'sub_rate': sub_rate,
    }


def main(args):
    assert args.split in ['test', 'dev'], f'Split must be either test or dev (got {args.split})'
    if args.dataset == 'rev16':
        assert args.split == 'test', 'Split must be test for rev16'

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    args.config = checkpoint['config']

    if args.disable_flash_attention:
        args.config.model.flash_attn = False

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size())
    model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()

    decoder = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=model.decoder.num_classes - 1)
    data = datasets_functions[args.dataset](args.split)
    print(f'Dataset {args.dataset} ({args.split}): {len(data)} records')

    beamsearch = None
    if args.beamsearch:
        beamsearch = lib.load_beamsearch(
            path=lib.paths.checkpoints.lm,
            alpha=args.__dict__.get('lm_alpha', 0.45),
            beta=args.__dict__.get('lm_beta', 1.53),
            prune_less_than_val=args.__dict__.get('lm_prune_less_than_val', 3.17),
            top_am_threshold=args.__dict__.get('lm_top_am_threshold', -6),
        )
    beams = args.__dict__.get('lm_eval_beams', 20)

    eval_fn = dynamic_eval if not args.awmc else AWMC
    adapt_overlap = args.adapt_overlap if args.adapt_overlap is not None else args.overlap
    if adapt_overlap != args.overlap:
        print(f'Using adapt_overlap={adapt_overlap} for adaptation (eval overlap={args.overlap})')

    original_model_params = [p.clone().detach().cpu() for p in model.parameters()]

    def restore_original_params():
        for p, u in zip(model.parameters(), original_model_params):
            p.data = u.data.to(p.device)

    def transcribe_logits(logits):
        if beamsearch is None:
            out_text = decoder(torch.as_tensor(logits))
        else:
            run_beam_search = beamsearch(log_probs=logits, beam_width=beams)
            run_beam_search.run_search(use_tqdm=True)
            out_text = run_beam_search.return_text(idx=0)
        return normalize(out_text).lower()

    baseline_args = copy.copy(args)
    baseline_args.epochs = 0

    def evaluate_records(records, eval_args, label):
        preds, golds, per_record = [], [], []
        for rec in tqdm(records, total=len(records), desc=label, leave=False):
            audio_spec, gold_text = rec['process_fn'](rec)
            logits = eval_fn(
                eval_args,
                model,
                audio_spec,
                args.seq_len,
                args.overlap,
                tokenizer,
                use_tqdm=False,
                beam_search_fn=beamsearch,
            )
            pred = transcribe_logits(logits)
            preds.append(pred)
            golds.append(gold_text)
            per_record.append({'id': rec['id'], 'prediction': pred, 'gold': gold_text})
        return score_texts(preds, golds), preds, golds, per_record

    all_specs = []
    for rec in tqdm(data, total=len(data), desc='concat-prep', leave=False):
        audio_spec, _ = rec['process_fn'](rec)
        all_specs.append(audio_spec)
    concat_spec = concatenate_specs(all_specs)
    all_ids = [rec['id'] for rec in data]
    print(f'Full concatenated adaptation spec shape={tuple(concat_spec.shape)}')
    print(f'Adapt ids ({len(all_ids)}): {all_ids}')

    all_repeat_scores = []

    for repeat in range(args.repeats):
        print(f'\n=== Repeat {repeat + 1}/{args.repeats} ===')
        restore_original_params()

        baseline_scores, baseline_preds, baseline_golds, baseline_per_record = evaluate_records(data, baseline_args, 'baseline eval')
        print(f'Baseline WER = {baseline_scores["wer"]}')

        updated_parameters = adapt_on_concat_only(
            args,
            model,
            concat_spec,
            tokenizer,
            beamsearch=beamsearch,
            adapt_overlap=adapt_overlap,
        )
        for p, u in zip(model.parameters(), updated_parameters):
            p.data = u.data.to(p.device)

        adapted_scores, adapted_preds, adapted_golds, adapted_per_record = evaluate_records(data, baseline_args, 'adapted eval')
        print(f'Adapted WER = {adapted_scores["wer"]}')
        print(f'Delta = {adapted_scores["wer"] - baseline_scores["wer"]:+.6f}')

        repeat_results = {
            'dataset': args.dataset,
            'split': args.split,
            'repeat': f'{repeat + 1}/{args.repeats}',
            'adapt_ids': all_ids,
            'adapt_num_records': len(all_ids),
            'concat_spec_shape': tuple(concat_spec.shape),
            'concat_total_frames': int(concat_spec.shape[-1]),
            'baseline': baseline_scores,
            'adapted': adapted_scores,
            'delta_wer': adapted_scores['wer'] - baseline_scores['wer'],
            'baseline_model_output': baseline_preds,
            'model_output': adapted_preds,
            'gold': adapted_golds,
            'baseline_per_record': baseline_per_record,
            'adapted_per_record': adapted_per_record,
            'args_dict': vars(args),
        }

        if args.save_path != '':
            save_path = args.save_path
            if save_path.endswith('.pkl'):
                save_path = save_path.replace('.pkl', f'_{repeat + 1}.pkl')
            else:
                save_path = save_path + f'_{repeat + 1}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(repeat_results, f)
            print(f'Saved to {save_path}')

        all_repeat_scores.append(adapted_scores['wer'])
        restore_original_params()

    if len(all_repeat_scores) > 0:
        avg = sum(all_repeat_scores) / len(all_repeat_scores)
        print(f'Average adapted WER across repeats: {avg}')
        return avg
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='earnings22', choices=datasets_functions.keys(), required=True)
    parser.add_argument('--repeats', '-r', type=int, default=1, help='Number of times to repeat the evaluation')
    parser.add_argument('--save_path', '-s', type=str, default='', help='path to save')
    parser.add_argument('--adapt_overlap', '-ao', type=int, default=None, help='Overlap used during adaptation passes only. If unset, adaptation uses --overlap.')

    args = lib.apply_args(parser)
    main(args)
