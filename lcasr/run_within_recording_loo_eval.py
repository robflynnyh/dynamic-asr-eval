import torch, argparse, lcasr
from tqdm import tqdm
from lcasr.utils.general import load_model
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.eval.wer import word_error_rate_detail
from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()
import pickle

import lib
from lib import dynamic_eval, AWMC, prepare_chunks

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

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_config = checkpoint['config']
    args.config = model_config

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
    print(f'LOO chunking: seq_len={args.seq_len}, overlap={args.overlap}')

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

    original_model_params = [p.clone().detach().cpu() for p in model.parameters()]

    downsampling_factor = args.config['model']['subsampling_factor']
    vocab_size_plus = tokenizer.vocab_size() + 1

    def restore_original_params():
        for p, u in zip(model.parameters(), original_model_params):
            p.data = u.data.to(p.device)

    def transcribe_from_logits(logits):
        if beamsearch is None:
            out_text = decoder(torch.as_tensor(logits))
        else:
            run_beam_search = beamsearch(log_probs=logits, beam_width=beams)
            run_beam_search.run_search(use_tqdm=True)
            out_text = run_beam_search.return_text(idx=0)
        return normalize(out_text).lower()

    def single_pass_logits(audio_spec):
        audio_chunk = audio_spec.clone().to(model.device)
        model.eval()
        with torch.no_grad():
            out = model(audio_signal=audio_chunk)
        logits = torch.exp(out['final_posteriors'][0].detach().cpu())
        return torch.log(logits).numpy()

    def loo_eval(audio_spec):
        spec_n = audio_spec.shape[-1]
        chunks, chunk_keys = prepare_chunks(audio_spec, args.seq_len, args.overlap)
        chunk_keys = sorted(chunk_keys)
        n_chunks = len(chunk_keys)

        if n_chunks <= 1:
            print(f'  Only {n_chunks} chunk(s) at seq_len={args.seq_len}; falling back to single no-adapt forward pass.')
            return single_pass_logits(audio_spec), {'n_chunks': n_chunks, 'mode': 'fallback_single_pass'}

        print(f'  {n_chunks} chunks -> {n_chunks} adaptations + {n_chunks * (n_chunks - 1)} forward passes')

        all_logits = torch.zeros((1, spec_n // downsampling_factor + args.seq_len, vocab_size_plus))
        logit_count = torch.zeros_like(all_logits)

        adapt_pbar = tqdm(chunk_keys, desc='  adapt-i', leave=False)
        for adapt_i in adapt_pbar:
            restore_original_params()
            adapt_chunk = chunks[adapt_i]

            _, updated_params = eval_fn(
                args,
                model,
                adapt_chunk,
                args.seq_len,
                0,
                tokenizer,
                beam_search_fn=beamsearch,
                return_params=True,
            )
            for p, u in zip(model.parameters(), updated_params):
                p.data = u.data.to(p.device)

            model.eval()
            for eval_j in chunk_keys:
                if eval_j == adapt_i:
                    continue
                audio_chunk = chunks[eval_j].clone().to(model.device)
                with torch.no_grad():
                    out = model(audio_signal=audio_chunk)
                logits = torch.exp(out['final_posteriors'][0].detach().cpu())
                ds_len = logits.shape[-2]
                ds_pos = eval_j // downsampling_factor
                all_logits[:, ds_pos:ds_pos + ds_len, :] += logits
                logit_count[:, ds_pos:ds_pos + ds_len, :] += 1

        restore_original_params()

        nonzero_mask = logit_count.sum(dim=-1) != 0
        nz_idx = nonzero_mask[0].nonzero(as_tuple=False).squeeze(-1)
        if nz_idx.numel() == 0:
            raise RuntimeError('LOO stitching produced no coverage at any position.')
        first, last = nz_idx[0].item(), nz_idx[-1].item()
        expected_span = last - first + 1
        if nz_idx.numel() != expected_span:
            gap = expected_span - nz_idx.numel()
            raise RuntimeError(f'LOO stitching has {gap} uncovered position(s) inside covered span [{first}, {last}].')

        all_logits = all_logits[nonzero_mask].reshape(1, -1, vocab_size_plus)
        logit_count = logit_count[nonzero_mask].reshape(1, -1, vocab_size_plus)

        avg_logits = all_logits / logit_count
        log_logits = torch.log(avg_logits)
        return log_logits.squeeze(0).numpy(), {'n_chunks': n_chunks, 'mode': 'loo'}

    for repeat in range(args.repeats):
        print(f'\n=== Repeat {repeat + 1}/{args.repeats} ===')
        all_texts, all_golds, meta = [], [], []

        for rec_idx in tqdm(range(len(data)), total=len(data), desc='records'):
            rec = data[rec_idx]
            print(f'\n-------\n{rec["id"]}\n-------')
            audio_spec, gold_text = rec['process_fn'](rec)
            stitched_logits, info = loo_eval(audio_spec)
            pred = transcribe_from_logits(stitched_logits)
            print(f'GOLD: {gold_text}')
            print(f'PRED: {pred}')
            all_texts.append(pred)
            all_golds.append(gold_text)
            meta.append({'id': rec['id'], **info})

        wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)
        print(f'\nRepeat {repeat + 1} WER: {wer}')

        if args.save_path != '':
            save_data = {
                'wer': wer,
                'words': words,
                'ins_rate': ins_rate,
                'del_rate': del_rate,
                'sub_rate': sub_rate,
                'model_output': all_texts,
                'gold': all_golds,
                'per_recording_meta': meta,
                'dataset': args.dataset,
                'args_dict': vars(args),
                'repeat': f'{repeat + 1}/{args.repeats}',
            }
            save_path = args.save_path
            if save_path.endswith('.pkl'):
                save_path = save_path.replace('.pkl', f'_{repeat + 1}.pkl')
            else:
                save_path = save_path + f'_{repeat + 1}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            print(f'Saved to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='earnings22', choices=datasets_functions.keys())
    parser.add_argument('--repeats', '-r', type=int, default=1, help='Number of times to repeat the evaluation')
    parser.add_argument('--save_path', '-s', type=str, default='', help='path to save')

    args = lib.apply_args(parser)
    main(args)


# Example:
# CUDA_VISIBLE_DEVICES="0" python run_within_recording_loo_eval.py \
#   -dfa -epochs 1 -seq 65536 -o 57344 -split test -d earnings22 -r 1 \
#   -s "./results/within_loo/earnings22-test.pkl" \
#   -kwargs optim_lr=0.00009 spec_augment_n_freq_masks=6 spec_augment_freq_mask_param=34 spec_augment_n_time_masks=0
