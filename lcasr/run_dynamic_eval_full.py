import argparse
import pickle
import time

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
from lib import AWMC, dynamic_eval, dynamic_eval_consistency_ctc_loss

normalize = EnglishTextNormalizer()


datasets_functions = {
    'earnings22': get_text_and_audio_earnings22,
    'chime6': get_text_and_audio_chime6,
    'tedlium': get_text_and_audio_tedlium,
    'rev16': get_text_and_audio_rev16,
}


def main(args):
    assert args.split in ['test', 'dev'], f'Split must be either test or dev (got {args.split})'
    if args.dataset == 'rev16':
        assert args.split == 'test', 'Split must be test for rev16'

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

    decoder = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=model.decoder.num_classes - 1)
    data = datasets_functions[args.dataset](args.split)

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

    if args.awmc:
        eval_fn = AWMC
    elif args.consistency:
        eval_fn = dynamic_eval_consistency_ctc_loss
    else:
        eval_fn = dynamic_eval

    avg_wers = []

    for repeat in range(args.repeats):
        all_texts, all_golds = [], []
        elapsed_times = []

        for rec in tqdm(range(len(data)), total=len(data)):
            print(f'Processing {rec + 1}/{len(data)}')
            print('\n-------\n' + data[rec]['id'] + '\n-------\n')

            audio_spec, gold_text = data[rec]['process_fn'](data[rec])

            stime = time.time()
            logits = eval_fn(
                args,
                model,
                audio_spec,
                args.seq_len,
                args.overlap,
                tokenizer,
                beam_search_fn=beamsearch,
            )
            etime = time.time()
            elapsed_times.append(etime - stime)

            if beamsearch is None:
                out_text = decoder(torch.as_tensor(logits))
            else:
                run_beam_search = beamsearch(log_probs=logits, beam_width=beams)
                run_beam_search.run_search(use_tqdm=True)
                out_text = run_beam_search.return_text(idx=0)

            out = normalize(out_text).lower()
            print(gold_text, '\n', out, '\n\n')

            all_texts.append(out)
            all_golds.append(gold_text)

        wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(
            hypotheses=all_texts,
            references=all_golds,
        )

        print(f'WER: {wer}')

        if args.log != '':
            with open(args.log, 'a') as f:
                f.write(f'{args.checkpoint}\t overlap: {args.overlap}\t seq_len: {args.seq_len}\t WER: {wer}\n')

        if args.save_path != '':
            save_data = {
                'wer': wer,
                'words': words,
                'ins_rate': ins_rate,
                'del_rate': del_rate,
                'sub_rate': sub_rate,
                'model_output': all_texts,
                'gold': all_golds,
                'elapsed_times': elapsed_times,
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

        avg_wers.append(wer)

    avg = sum(avg_wers) / len(avg_wers)
    print(f'Average WER: {avg}')
    return avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='earnings22', choices=datasets_functions.keys())
    parser.add_argument('--repeats', '-r', type=int, default=1, help='Number of times to repeat the evaluation')
    parser.add_argument('--save_path', '-s', type=str, default='', help='path to save')

    args = lib.apply_args(parser)
    main(args)
