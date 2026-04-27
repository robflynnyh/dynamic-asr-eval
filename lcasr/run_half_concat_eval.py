import argparse
import copy
import pickle
import random

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


def split_records(data, shuffle_splits, seed):
    indices = list(range(len(data)))
    if shuffle_splits:
        rng = random.Random(seed)
        rng.shuffle(indices)
    midpoint = len(indices) // 2
    if midpoint == 0:
        raise ValueError('Need at least 2 records to make a two-way split.')
    first = indices[:midpoint]
    second = indices[midpoint:]
    if len(second) == 0:
        raise ValueError('Second half is empty after split.')
    return first, second


def concatenate_specs(specs):
    if len(specs) == 0:
        raise ValueError('Cannot concatenate an empty list of spectrograms.')
    return torch.cat(specs, dim=-1)


def adapt_on_concat_only(args, model, concat_spec, tokenizer, beamsearch=None, adapt_overlap=None):
    """Adapt on a concatenated spectrogram without building final stitched logits.
    This avoids the large all_logits/logit_count allocation in lib.dynamic_eval when
    we only need updated parameters for subsequent held-out evaluation.
    """
    if args.awmc:
        _, updated_parameters = AWMC(
            args,
            model,
            concat_spec,
            args.seq_len,
            adapt_overlap,
            tokenizer,
            use_tqdm=False,
            beam_search_fn=beamsearch,
            return_params=True,
        )
        return updated_parameters

    spec_n = concat_spec.shape[-1]
    downsampling_factor = args.config['model']['subsampling_factor']
    seq_len = args.seq_len if args.seq_len != -1 else args.config['audio_chunking']['size']

    spec_augment_config = lib.get_specaugment_config_from_args(args)
    random_noise = args.__dict__.get('random_noise', 0.0)
    lr_args = lib.get_lr_args_from_args(args)
    frame_shuffle_args = lib.get_frame_shuffle_config_from_args(args)
    entropy_args = {k.replace('entropy_augmentation_', ''): v for k, v in args.__dict__.items() if k.startswith('entropy_augmentation_')}
    cutout_args = lib.get_cutout_params_from_args(args, seq_len)
    num_negatives = 1

    original_model_params = [p.clone().detach().cpu() for p in model.parameters()]

    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes - 1, reduction='sum')
    optimizer = lib.madgrad.MADGRAD(model.parameters(), **lr_args)
    decoder = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=model.decoder.num_classes - 1)
    augmentation = lib.SpecAugment(**spec_augment_config)

    if seq_len > spec_n:
        seq_len, adapt_overlap = spec_n, 0
    else:
        adapt_overlap = adapt_overlap if adapt_overlap != -1 else args.config['audio_chunking']['overlap']

    assert args.config['training'].get('max_seq_len', 0) == 0, 'caching is not used anymore'
    assert adapt_overlap / downsampling_factor == adapt_overlap // downsampling_factor, 'Overlap must be a multiple of the downsampling factor'

    epochs = args.__dict__.get('epochs', 1)
    shuffle = args.__dict__.get('shuffle', False)
    beams = args.__dict__.get('lm_tta_beams', 3)

    model.eval()
    training_data, training_keys = lib.prepare_chunks(concat_spec, seq_len, adapt_overlap)
    print(f'Adapt-only pass on concatenated spec: {len(training_keys)} chunks, seq_len={seq_len}, overlap={adapt_overlap}')

    for epoch in range(epochs):
        print(f'Adapt epoch {epoch + 1} / {epochs}')
        cur_keys = list(training_keys)
        cur_keys = random.sample(cur_keys, len(cur_keys)) if shuffle else cur_keys
        pbar = tqdm(cur_keys, desc='adapt-only', leave=False)
        for i in pbar:
            audio_chunk = training_data[i].clone()
            audio_chunk = audio_chunk.repeat(num_negatives + 1, 1, 1)
            audio_chunk[:num_negatives] = augmentation(audio_chunk[:num_negatives])
            audio_chunk[:num_negatives] = lib.frame_shuffle(audio_chunk[:num_negatives], **frame_shuffle_args)
            audio_chunk[:num_negatives] = lib.add_random_noise(audio_chunk[:num_negatives], noise_factor=random_noise)
            audio_chunk[:num_negatives] = lib.cutout(audio_chunk[:num_negatives], **cutout_args)
            audio_chunk[:num_negatives] = lib.entropy_augmentation(audio_chunk[:num_negatives], model, **entropy_args)

            audio_chunk = audio_chunk.to(model.device)
            out = model(audio_signal=audio_chunk)

            if beamsearch is None or beams == 0:
                pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu())
            else:
                run_beam_search = beamsearch(log_probs=out['final_posteriors'][-1].detach().cpu(), beam_width=beams)
                run_beam_search.run_search(use_tqdm=False)
                pseudo_targets = run_beam_search.return_text(idx=0)

            pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets)).unsqueeze(0).to(model.device).repeat(num_negatives, 1)
            augmented_outs = out['final_posteriors'][:num_negatives]
            n_tokens, batch_size = augmented_outs.shape[1], augmented_outs.shape[0]
            total_tokens_in_loss = n_tokens * batch_size
            loss = ctc_loss_fn(
                augmented_outs.transpose(0, 1),
                pseudo_targets,
                torch.LongTensor([n_tokens] * augmented_outs.shape[0]).to(model.device),
                torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0]).to(model.device),
            ) / total_tokens_in_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    updated_model_params = [p.clone().detach().cpu() for p in model.parameters()]
    for p, p_orig in zip(model.parameters(), original_model_params):
        p.data = p_orig.data.to(p.device)
    return updated_model_params


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

    decoder = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=model.decoder.num_classes-1)

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

    def transcribe_from_logits(logits, audio_spec):
        if beamsearch is None:
            out_text = decoder(torch.as_tensor(logits))
        else:
            run_beam_search = beamsearch(log_probs=logits, beam_width=beams)
            run_beam_search.run_search(use_tqdm=True)
            out_text = run_beam_search.return_text(idx=0)
        return normalize(out_text).lower()

    baseline_args = copy.copy(args)
    baseline_args.epochs = 0

    def evaluate_records(records, fold_name):
        preds, golds, per_record = [], [], []
        for rec in tqdm(records, total=len(records), desc=f'{fold_name} eval', leave=False):
            audio_spec, gold_text = rec['process_fn'](rec)
            logits = eval_fn(
                baseline_args,
                model,
                audio_spec,
                args.seq_len,
                args.overlap,
                tokenizer,
                use_tqdm=False,
                beam_search_fn=beamsearch,
            )
            pred = transcribe_from_logits(logits, audio_spec)
            preds.append(pred)
            golds.append(gold_text)
            per_record.append({'id': rec['id'], 'prediction': pred, 'gold': gold_text})
        return score_texts(preds, golds), preds, golds, per_record

    all_repeat_scores = []

    for repeat in range(args.repeats):
        print(f'\n=== Repeat {repeat + 1}/{args.repeats} ===')
        seed = args.split_seed + repeat
        first_idxs, second_idxs = split_records(data, args.shuffle_splits, seed)
        fold_pairs = [
            ('fold_1_to_2', first_idxs, second_idxs),
            ('fold_2_to_1', second_idxs, first_idxs),
        ]

        repeat_results = {
            'dataset': args.dataset,
            'split': args.split,
            'repeat': f'{repeat + 1}/{args.repeats}',
            'shuffle_used': args.shuffle,
            'shuffle_splits': args.shuffle_splits,
            'split_seed': seed,
            'folds': [],
            'args_dict': vars(args),
        }

        fold_wers = []
        baseline_wers = []

        for fold_name, adapt_idxs, eval_idxs in fold_pairs:
            restore_original_params()
            adapt_records = [data[idx] for idx in adapt_idxs]
            eval_records = [data[idx] for idx in eval_idxs]

            print(f'\n--- {fold_name} ---')
            print(f'Adapt ids ({len(adapt_records)}): {[rec["id"] for rec in adapt_records]}')
            print(f'Eval ids  ({len(eval_records)}): {[rec["id"] for rec in eval_records]}')

            adapt_specs = []
            for rec in tqdm(adapt_records, total=len(adapt_records), desc=f'{fold_name} adapt-prep', leave=False):
                audio_spec, _ = rec['process_fn'](rec)
                adapt_specs.append(audio_spec)
            concat_spec = concatenate_specs(adapt_specs)
            print(f'{fold_name}: concatenated adaptation spec shape={tuple(concat_spec.shape)}')

            baseline_scores, baseline_preds, baseline_golds, baseline_per_record = evaluate_records(eval_records, f'{fold_name} baseline')
            print(f'{fold_name}: baseline WER on held-out half = {baseline_scores["wer"]}')

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

            adapted_scores, adapted_preds, adapted_golds, adapted_per_record = evaluate_records(eval_records, fold_name)
            print(f'{fold_name}: adapted WER on held-out half = {adapted_scores["wer"]}')
            print(f'{fold_name}: delta = {adapted_scores["wer"] - baseline_scores["wer"]:+.6f}')

            fold_result = {
                'name': fold_name,
                'adapt_ids': [rec['id'] for rec in adapt_records],
                'eval_ids': [rec['id'] for rec in eval_records],
                'adapt_num_records': len(adapt_records),
                'eval_num_records': len(eval_records),
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
            }
            repeat_results['folds'].append(fold_result)
            fold_wers.append(adapted_scores['wer'])
            baseline_wers.append(baseline_scores['wer'])

        repeat_results['aggregate'] = {
            'baseline_mean_wer': sum(baseline_wers) / len(baseline_wers),
            'adapted_mean_wer': sum(fold_wers) / len(fold_wers),
            'delta_mean_wer': (sum(fold_wers) / len(fold_wers)) - (sum(baseline_wers) / len(baseline_wers)),
        }
        print(
            'Repeat summary: '
            f'baseline_mean_wer={repeat_results["aggregate"]["baseline_mean_wer"]} '
            f'adapted_mean_wer={repeat_results["aggregate"]["adapted_mean_wer"]} '
            f'delta={repeat_results["aggregate"]["delta_mean_wer"]:+.6f}'
        )

        if args.save_path != '':
            save_path = args.save_path
            if save_path.endswith('.pkl'):
                save_path = save_path.replace('.pkl', f'_{repeat + 1}.pkl')
            else:
                save_path = save_path + f'_{repeat + 1}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(repeat_results, f)
            print(f'Saved to {save_path}')

        all_repeat_scores.append(repeat_results['aggregate']['adapted_mean_wer'])
        restore_original_params()

    if len(all_repeat_scores) > 0:
        avg = sum(all_repeat_scores) / len(all_repeat_scores)
        print(f'Average adapted mean WER across repeats: {avg}')
        return avg
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='earnings22', choices=datasets_functions.keys(), required=True)
    parser.add_argument('--repeats', '-r', type=int, default=1, help='Number of times to repeat the evaluation')
    parser.add_argument('--save_path', '-s', type=str, default='', help='path to save')
    parser.add_argument('--adapt_overlap', '-ao', type=int, default=None, help='Overlap used during adaptation passes only. If unset, adaptation uses --overlap.')
    parser.add_argument('--split_seed', type=int, default=0, help='Base seed for repeat-wise half splits when shuffling is enabled.')
    parser.add_argument('--shuffle_splits', action='store_true', help='Shuffle record order before splitting into halves on each repeat.')

    args = lib.apply_args(parser)
    main(args)
