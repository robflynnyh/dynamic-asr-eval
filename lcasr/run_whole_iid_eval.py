import argparse
import copy
import pickle
import random

import lcasr
import torch
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

import lib
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.utils.general import load_model
from lib import AWMC, dynamic_eval
from run_whole_concat_eval import datasets_functions, score_texts

normalize = EnglishTextNormalizer()


def build_iid_chunk_pool(records, seq_len, adapt_overlap):
    chunks = []
    chunks_per_record = {}
    for rec in tqdm(records, total=len(records), desc='iid-chunk-prep', leave=False):
        audio_spec, _ = rec['process_fn'](rec)
        training_data, training_keys = lib.prepare_chunks(audio_spec, seq_len, adapt_overlap)
        chunks_per_record[rec['id']] = len(training_keys)
        for key in training_keys:
            chunks.append({
                'record_id': rec['id'],
                'chunk_key': key,
                'audio': training_data[key],
            })
    return chunks, chunks_per_record


def adapt_on_iid_chunks_only(args, model, chunk_pool, tokenizer, beamsearch=None):
    """Adapt on a shuffled pool of per-record chunks.

    Unlike whole-concat adaptation, chunks are created independently per
    recording, so no adaptation sample spans a recording boundary.
    """
    if args.awmc:
        raise NotImplementedError('AWMC is not implemented for whole-iid pooled chunk adaptation.')
    if len(chunk_pool) == 0:
        raise ValueError('Cannot adapt with an empty chunk pool.')

    spec_augment_config = lib.get_specaugment_config_from_args(args)
    random_noise = args.__dict__.get('random_noise', 0.0)
    lr_args = lib.get_lr_args_from_args(args)
    frame_shuffle_args = lib.get_frame_shuffle_config_from_args(args)
    entropy_args = {
        k.replace('entropy_augmentation_', ''): v
        for k, v in args.__dict__.items()
        if k.startswith('entropy_augmentation_')
    }
    cutout_args = lib.get_cutout_params_from_args(args, args.seq_len)
    num_negatives = 1

    original_model_params = [p.clone().detach().cpu() for p in model.parameters()]

    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes - 1, reduction='sum')
    optimizer = lib.madgrad.MADGRAD(model.parameters(), **lr_args)
    decoder = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=model.decoder.num_classes - 1)
    augmentation = lib.SpecAugment(**spec_augment_config)
    beams = args.__dict__.get('lm_tta_beams', 3)
    epochs = args.__dict__.get('epochs', 1)
    shuffle = args.__dict__.get('shuffle', False)

    model.eval()
    print(f'Adapt-only pass on iid chunk pool: {len(chunk_pool)} chunks')

    for epoch in range(epochs):
        print(f'Adapt epoch {epoch + 1} / {epochs}')
        epoch_chunks = list(chunk_pool)
        if shuffle:
            epoch_chunks = random.sample(epoch_chunks, len(epoch_chunks))
        pbar = tqdm(epoch_chunks, desc='adapt-iid', leave=False)
        for chunk in pbar:
            audio_chunk = chunk['audio'].clone()
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

            pseudo_targets = (
                torch.LongTensor(tokenizer.encode(pseudo_targets))
                .unsqueeze(0)
                .to(model.device)
                .repeat(num_negatives, 1)
            )
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

    seq_len = args.seq_len if args.seq_len != -1 else args.config['audio_chunking']['size']
    if adapt_overlap == -1:
        adapt_overlap = args.config['audio_chunking']['overlap']
    downsampling_factor = args.config['model']['subsampling_factor']
    assert adapt_overlap / downsampling_factor == adapt_overlap // downsampling_factor, 'Overlap must be a multiple of the downsampling factor'

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

    chunk_pool, chunks_per_record = build_iid_chunk_pool(data, seq_len, adapt_overlap)
    all_ids = [rec['id'] for rec in data]
    print(f'IID adaptation pool chunks={len(chunk_pool)}, seq_len={seq_len}, adapt_overlap={adapt_overlap}')
    print(f'Adapt ids ({len(all_ids)}): {all_ids}')

    all_repeat_scores = []

    for repeat in range(args.repeats):
        print(f'\n=== Repeat {repeat + 1}/{args.repeats} ===')
        restore_original_params()

        baseline_scores, baseline_preds, baseline_golds, baseline_per_record = evaluate_records(data, baseline_args, 'baseline eval')
        print(f'Baseline WER = {baseline_scores["wer"]}')

        updated_parameters = adapt_on_iid_chunks_only(
            args,
            model,
            chunk_pool,
            tokenizer,
            beamsearch=beamsearch,
        )
        for p, u in zip(model.parameters(), updated_parameters):
            p.data = u.data.to(p.device)

        adapted_scores, adapted_preds, adapted_golds, adapted_per_record = evaluate_records(data, baseline_args, 'adapted eval')
        print(f'Adapted WER = {adapted_scores["wer"]}')
        print(f'Delta = {adapted_scores["wer"] - baseline_scores["wer"]:+.6f}')

        repeat_results = {
            'dataset': args.dataset,
            'split': args.split,
            'adaptation_mode': 'whole_iid_chunks',
            'repeat': f'{repeat + 1}/{args.repeats}',
            'adapt_ids': all_ids,
            'adapt_num_records': len(all_ids),
            'adapt_num_chunks': len(chunk_pool),
            'chunks_per_record': chunks_per_record,
            'adapt_seq_len': seq_len,
            'adapt_overlap': adapt_overlap,
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
