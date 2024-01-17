from omegaconf import OmegaConf
import os
#paths = OmegaConf.load('paths.yaml')
# use absolute path
paths = OmegaConf.load(os.path.join(os.path.dirname(__file__), '../paths.yaml'))
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Callable
import torch.optim as optim

from lcasr.utils.augmentation import SpecAugment
from lcasr.decoding.greedy import GreedyCTCDecoder
import madgrad, random
from soft_dtw_cuda import SoftDTW
from einops import rearrange
from lcasr.decoding import ctc_beam_search as beam_search
from lming.utils import general
import lcasr
from functools import partial
from matplotlib import pyplot as plt
from lcasr.models import nemo_sconformer

def load_model(path:str):
    model = nemo_sconformer.load_from_old_state_dict(
        path = path,
        instance = nemo_sconformer.load_defaul_instance(),
    )
    return model

def load_tokenizer(path:str):
    return nemo_sconformer.load_tokenizer(model_path = path)

def load_beamsearch(path):
    checkpoint = torch.load(path, map_location='cpu')
    checkpoint['model'] = general.convert_from_ddp(checkpoint['model'])
    model_config = checkpoint['config']
    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = general.load_model(config=model_config, vocab_size=tokenizer.vocab_size())
    model.load_state_dict(checkpoint['model'], strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()
    language_model = beam_search.LanguageModel(
        model = model,
        bos_id = tokenizer.bos_id(),
        device = device,
    )
    BeamSearch = partial(
        beam_search.BeamSearch,
        language_model = language_model,
        tokenizer = tokenizer,
        beam_width = 3,
        blank_id = tokenizer.vocab_size(),
        alpha = 0.45,
        beta = 1.53,
        debug = False,
        prune_less_than_val = 3.17,
        top_am_threshold = -6,
    )
    return BeamSearch

def replace_with_frame(spec):
    for i, s in enumerate(spec):
        random_index = random.randint(0, spec.shape[-1])
        # set all frames to random frame
        spec[i] = spec[i, :, :] * 0 + spec[i, :, random_index, None]
    return spec


class tokenizer_wrapper():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def encode(self, text):
        return self.tokenizer.text_to_ids(text)
    def decode(self, ids):
        return self.tokenizer.ids_to_text(ids)


def dynamic_eval_ctc_loss(
        args, 
        model:nn.Module, 
        utterances:torch.Tensor, 
        seq_len:int, 
        overlap:int, 
        tokenizer, 
        use_tqdm=True,
        optim:optim.Optimizer=madgrad.MADGRAD,
        num_negatives:int=2,
        lr_args:dict={'lr':0},#9e-5},
        optimizer_state:dict=None,
        spec_augment_config={
            'n_time_masks': 2,
            'n_freq_masks': 3,
            'freq_mask_param': 42,
            'time_mask_param': -1,
            'min_p': 0.05,
            'zero_masking': False,
        },
        beam_search_fn:Callable=None,
    ):


    # create copy of model parameters that are not updated
    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach() for p in original_model_params]
 
    
    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder._num_classes-1, reduction='sum')
    optimizer = optim(model.parameters(), **lr_args)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    
    decoder = GreedyCTCDecoder(tokenizer = tokenizer_wrapper(tokenizer), blank_id = model.decoder._num_classes-1)
    augmentation = SpecAugment(**spec_augment_config)

    losses = []
    
    model.eval()
    #
    print(f'Number of training examples: {len(utterances)}')
    for epoch in range(args.__dict__.get('epochs', 1)):
        print(f'Epoch {epoch + 1} / {args.__dict__.get("epochs", 1)}')
        model_outputs = {}
        indexes = list(range(len(utterances)))
        indexes = random.sample(indexes, len(indexes)) if args.__dict__.get('shuffle', False) else indexes
        print(indexes[0])
        pbar = tqdm(indexes) if use_tqdm else indexes
        
        for idx in pbar:
            utt = utterances[idx]
            audio_chunk = utt['waveform']
            length = audio_chunk.shape[-1]
            audio_chunk = audio_chunk.to(model.device)
            audio_chunk, length = model.preprocessor(audio_chunk, torch.LongTensor([length]).to(model.device))
            audio_chunk = audio_chunk.repeat(num_negatives+1, 1, 1).contiguous() # [B, C, T]

            audio_chunk = audio_chunk.to(model.device)
            out = model(audio_signal = audio_chunk)

            pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu())
            noisy_predictions = decoder(out['final_posteriors'][0].detach().cpu())
            print(f'Pseudo targets: {pseudo_targets}')
            print(f'Noisy predictions: {noisy_predictions}')
            print('\n--\n')
            pseudo_targets = torch.LongTensor(tokenizer.text_to_ids(pseudo_targets)).unsqueeze(0).to(model.device).repeat(num_negatives, 1)
            augmented_outs = out['final_posteriors'][:num_negatives]            
            
            N, B = augmented_outs.shape[1], augmented_outs.shape[0]
            total_tokens_in_loss = N * B

            loss = ctc_loss_fn(augmented_outs.transpose(0, 1), pseudo_targets, torch.LongTensor([N] * augmented_outs.shape[0]).to(model.device), torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0]).to(model.device)) / total_tokens_in_loss
            losses.append(loss.item())

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            plt.plot(losses)
            plt.savefig('loss.png')
            plt.close()

            probs = out['final_posteriors'][-1].detach().cpu()
            utterances[idx]['probs'] = probs

    # reset model parameters
    for p, p_orig in zip(model.parameters(), original_model_params):
        p.data = p_orig.data


    return utterances



dynamic_eval = dynamic_eval_ctc_loss


'''
shared functions between scripts
'''
def apply_args(parser):
    parser.add_argument('-c', '--checkpoint', type=str, default='', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=2048, help='sequence length')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-nv', '--not_verbose', action='store_true', help='verbose')
    parser.add_argument('-log', '--log', type=str, default='')
    parser.add_argument('-shuffle', '--shuffle', action='store_true', help='shuffle')
    parser.add_argument('-epochs', '--epochs', type=int, default=1, help='epochs')
    parser.add_argument('-beamsearch', '--beamsearch', action='store_true', help='use beam search')

    args = parser.parse_args()
    args.verbose = not args.not_verbose
    if args.checkpoint == '':
        args.checkpoint = paths.checkpoints.lcasr_nemo

    # get absolute path of ./
    args.tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.model")
    return args
