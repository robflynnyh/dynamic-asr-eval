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

def get_specaugment_config_from_args(args):
    spec_augment_args = {k.replace('spec_augment_', ''):v for k,v in args.__dict__.items() if k.startswith('spec_augment')}
    spec_augment_config = {
        'n_time_masks': spec_augment_args.get('n_time_masks', 2),
        'n_freq_masks': spec_augment_args.get('n_freq_masks', 3),
        'freq_mask_param': spec_augment_args.get('freq_mask_param', 42),
        'time_mask_param': spec_augment_args.get('time_mask_param', -1),
        'min_p': spec_augment_args.get('min_p', 0.05),
        'zero_masking': spec_augment_args.get('zero_masking', False),
    }
    return spec_augment_config

def get_lr_args_from_args(args):
    lr_args = {k.replace('optim_', ''):v for k,v in args.__dict__.items() if k.startswith('lr_')}
    lr_args['lr'] = lr_args.get('lr', 9e-5)
    return lr_args


def prepare_chunks(spec, seq_len, overlap):
    spec_n = spec.shape[-1]
    last_ulen, kill_next = None, False
    training_data = {}
    for i in range(0, spec_n, seq_len-overlap):
        audio_chunk = spec[:, :, i:i+seq_len] # [B, C, T]
        u_len = audio_chunk.shape[-1]
        if kill_next:
            break
        elif last_ulen != None and u_len < last_ulen:
            kill_next = True
        last_ulen = u_len
        training_data[i] = audio_chunk
    return training_data, list(training_data.keys())

def dynamic_eval_ctc_loss(
        args, 
        model:nn.Module, 
        spec:torch.Tensor, 
        seq_len:int, 
        overlap:int, 
        tokenizer, 
        use_tqdm=True,
        optim:optim.Optimizer=madgrad.MADGRAD,
        optimizer_state:dict=None,
        beam_search_fn:Callable=None,
    ):

    spec_augment_config = get_specaugment_config_from_args(args)
    lr_args = get_lr_args_from_args(args)
    num_negatives = args.__dict__.get('num_negatives', 1)
    
    spec_n = spec.shape[-1]
    downsampling_factor = args.config['model']['subsampling_factor']
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']

    # create copy of model parameters that are not updated
    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach() for p in original_model_params]
 
    
    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='sum')
    optimizer = optim(model.parameters(), **lr_args)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = model.decoder.num_classes-1)
    augmentation = SpecAugment(**spec_augment_config)

    if seq_len > spec_n:
        seq_len, overlap = spec_n, 0
    else:
        overlap = overlap if overlap != -1 else args.config['audio_chunking']['overlap']

    assert args.config['training'].get("max_seq_len", 0) == 0, 'caching is not used anymore'
    assert overlap / downsampling_factor == overlap // downsampling_factor, 'Overlap must be a multiple of the downsampling factor'
    print(f'Using seq_len: {seq_len} and overlap: {overlap}')

    all_logits, logit_count = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1)), torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1))
    
    model.eval() # don't update batchrenorm
    training_data, training_keys = prepare_chunks(spec, seq_len, overlap)
    for epoch in range(args.__dict__.get('epochs', 1)):
        print(f'Epoch {epoch + 1} / {args.__dict__.get("epochs", 1)}')
        training_keys = list(training_data.keys())
        training_keys = random.sample(training_keys, len(training_keys)) if args.__dict__.get('shuffle', False) else training_keys
      
        pbar = tqdm(training_keys) if use_tqdm else training_keys
        for i in pbar:
            audio_chunk = training_data[i].clone()
            audio_chunk = audio_chunk.repeat(num_negatives+1, 1, 1) # [B, C, T]
            audio_chunk[:num_negatives] = augmentation(audio_chunk[:num_negatives]) # apply augmentation to 2 of the 3 copies
 

            u_len = audio_chunk.shape[-1]
            audio_chunk = audio_chunk.to(model.device)
            out = model(audio_signal = audio_chunk)

            if beam_search_fn is None:
                pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu())
            else:
                beam_search = beam_search_fn(log_probs = out['final_posteriors'][-1].detach().cpu())
                beam_search.run_search(use_tqdm = True)
                pseudo_targets = beam_search.return_text(idx = 0)

            noisy_predictions = decoder(out['final_posteriors'][0].detach().cpu())
            print(f'Pseudo targets: {pseudo_targets}')
            print(f'Noisy predictions: {noisy_predictions}')
            print('\n--\n')
            pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets)).unsqueeze(0).to(model.device).repeat(num_negatives, 1)
            augmented_outs = out['final_posteriors'][:num_negatives]            
            
            N, B = augmented_outs.shape[1], augmented_outs.shape[0]
            total_tokens_in_loss = N * B
 
            loss = ctc_loss_fn(augmented_outs.transpose(0, 1), pseudo_targets, torch.LongTensor([N] * augmented_outs.shape[0]).to(model.device), torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0]).to(model.device)) / total_tokens_in_loss

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8) # add clip value to args
            optimizer.step()
        
    model.eval()
    training_data, training_keys = prepare_chunks(spec, seq_len, overlap)
    model_outputs = {}
    pbar = tqdm(training_keys) if use_tqdm else training_keys
    for i in pbar:
        audio_chunk = training_data[i].clone()
        u_len = audio_chunk.shape[-1]
        audio_chunk = audio_chunk.to(model.device)
        with torch.no_grad(): out = model(audio_signal = audio_chunk)
        logits = out['final_posteriors'][0].detach().cpu()
        logits = torch.exp(logits) # convert to prob
        ds_len = logits.shape[-2]
        ratio = u_len / ds_len
        overlap_ds = int(overlap / ratio)
        model_outputs[i] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}
    model.train()


           
    logit_position = 0
    for i in sorted(list(model_outputs.keys())):
        logits, ds_len, overlap_ds = model_outputs[i]['logits'], model_outputs[i]['ds_len'], model_outputs[i]['overlap_ds']
        logit_position -= overlap_ds if i != 0 else 0
        logit_count[:, logit_position:logit_position+ds_len, :] += 1
        all_logits[:, logit_position:logit_position+ds_len, :] += logits
        logit_position += ds_len 

    B,N,C = all_logits.shape
    all_logits = all_logits[logit_count.sum(dim=-1) != 0]
    all_logits = all_logits.reshape(B,-1,C)
    logit_count = logit_count[logit_count.sum(dim=-1) != 0]
    logit_count = logit_count.reshape(B,-1,C)
    logits = all_logits / logit_count
    logits = torch.log(logits) # convert to log 

    # reset model parameters
    for p, p_orig in zip(model.parameters(), original_model_params):
        p.data = p_orig.data


    return logits.squeeze(0).numpy()


'''
def dynamic_eval_ctc_loss(
        args, 
        model:nn.Module, 
        spec:torch.Tensor, 
        seq_len:int, 
        overlap:int, 
        tokenizer, 
        use_tqdm=True,
        optim:optim.Optimizer=madgrad.MADGRAD,
        num_negatives:int=2,
        optimizer_state:dict=None,
        beam_search_fn:Callable=None,
    ):

    spec_augment_config = get_specaugment_config_from_args(args)
    lr_args = get_lr_args_from_args(args)
    
    spec_n = spec.shape[-1]
    downsampling_factor = args.config['model']['subsampling_factor']
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']

    # create copy of model parameters that are not updated
    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach() for p in original_model_params]

 
    
    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='sum')
    optimizer = optim(model.parameters(), **lr_args)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = model.decoder.num_classes-1)
    augmentation = SpecAugment(**spec_augment_config)

    if seq_len > spec_n:
        seq_len, overlap = spec_n, 0
    else:
        overlap = overlap if overlap != -1 else args.config['audio_chunking']['overlap']

    assert args.config['training'].get("max_seq_len", 0) == 0, 'caching is not used anymore'
    assert overlap / downsampling_factor == overlap // downsampling_factor, 'Overlap must be a multiple of the downsampling factor'
    print(f'Using seq_len: {seq_len} and overlap: {overlap}')

    all_logits, logit_count = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1)), torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1))
    losses = []

    training_data, training_keys = prepare_chunks(spec, seq_len, 0) # no overlap for training (faster)

    for epoch in range(args.__dict__.get('epochs', 1)):
        print(f'Epoch {epoch + 1} / {args.__dict__.get("epochs", 1)}')
        training_keys = list(training_data.keys())
        training_keys = random.sample(training_keys, len(training_keys)) if args.__dict__.get('shuffle', False) else training_keys
        model_outputs = {}
        pbar = tqdm(training_keys) if use_tqdm else training_keys
        for i in pbar:
            audio_chunk = training_data[i].clone()
            audio_chunk = audio_chunk.repeat(num_negatives+1, 1, 1) # [B, C, T]
            audio_chunk[:num_negatives] = augmentation(audio_chunk[:num_negatives]) # apply augmentation to 2 of the 3 copies

            u_len = audio_chunk.shape[-1]
            audio_chunk = audio_chunk.to(model.device)
            out = model(audio_signal = audio_chunk)

            if beam_search_fn is None:
                pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu())
            else:
                beam_search = beam_search_fn(log_probs = out['final_posteriors'][-1].detach().cpu())
                beam_search.run_search(use_tqdm = True)
                pseudo_targets = beam_search.return_text(idx = 0)

            noisy_predictions = decoder(out['final_posteriors'][0].detach().cpu())
            print(f'Pseudo targets: {pseudo_targets}')
            print(f'Noisy predictions: {noisy_predictions}')
            print('\n--\n')
            pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets)).unsqueeze(0).to(model.device).repeat(num_negatives, 1)
            augmented_outs = out['final_posteriors'][:num_negatives]            
            
            N, B = augmented_outs.shape[1], augmented_outs.shape[0]
            total_tokens_in_loss = N * B

            loss = ctc_loss_fn(augmented_outs.transpose(0, 1), pseudo_targets, torch.LongTensor([N] * augmented_outs.shape[0]).to(model.device), torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0]).to(model.device)) / total_tokens_in_loss
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # plt.plot(losses)
            # plt.savefig('loss.png')
            # plt.close()
            logits = out['final_posteriors'][-1].detach().cpu()
            logits = torch.exp(logits) # convert to prob
            ds_len = logits.shape[-2]
            ratio = u_len / ds_len
            overlap_ds = int(overlap / ratio)
            model_outputs[i] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}

    #training_data, training_keys = prepare_chunks(spec, seq_len, 0)
    # model_outputs = {}
    # # pbar = tqdm(training_keys) if use_tqdm else training_keys
    # # print(training_keys)
    # with torch.no_grad():
    #     for i in pbar:
            
    #         audio_chunk = training_data[i].clone()
            
    #         u_len = audio_chunk.shape[-1]
    #         audio_chunk = audio_chunk.to(model.device)
    #         out = model(audio_signal = audio_chunk)

    #         logits = out['final_posteriors'][-1].detach().cpu()
    #         logits = torch.exp(logits) # convert to prob
    #         ds_len = logits.shape[-2]
    #         ratio = u_len / ds_len
    #         overlap_ds = int(overlap / ratio)
    #         model_outputs[i] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}

    logit_position = 0
    for i in sorted(list(model_outputs.keys())):
        logits, ds_len, overlap_ds = model_outputs[i]['logits'], model_outputs[i]['ds_len'], model_outputs[i]['overlap_ds']
        logit_position -= overlap_ds if i != 0 else 0
        logit_count[:, logit_position:logit_position+ds_len, :] += 1
        all_logits[:, logit_position:logit_position+ds_len, :] += logits
        logit_position += ds_len 

    B,N,C = all_logits.shape
    all_logits = all_logits[logit_count.sum(dim=-1) != 0]
    all_logits = all_logits.reshape(B,-1,C)
    logit_count = logit_count[logit_count.sum(dim=-1) != 0]
    logit_count = logit_count.reshape(B,-1,C)
    logits = all_logits / logit_count
    logits = torch.log(logits) # convert to log 

    # reset model parameters
    for p, p_orig in zip(model.parameters(), original_model_params):
        p.data = p_orig.data


    return logits.squeeze(0).numpy()
'''

dynamic_eval = dynamic_eval_ctc_loss


'''
shared functions between scripts
'''
def apply_args(parser):
    parser.add_argument('-c', '--checkpoint', type=str, default='', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-o', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-nv', '--not_verbose', action='store_true', help='verbose')
    parser.add_argument('-log', '--log', type=str, default='')
    parser.add_argument('-ds', '--dont_shuffle', action='store_true', help='dont shuffle')
    #parser.add_argument('-shuffle', '--shuffle', action='store_true', help='shuffle')
    parser.add_argument('-epochs', '--epochs', type=int, default=1, help='epochs')
    parser.add_argument('-dfa', '--disable_flash_attention', action='store_true', help='disable flash attention')
    parser.add_argument('-beamsearch', '--beamsearch', action='store_true', help='use beam search')
    parser.add_argument('-kwargs', '--kwargs', nargs='+', help='kwargs')

    args = parser.parse_args()

    if args.kwargs is None: args.kwargs = []
    for kwarg in args.kwargs:
        key, value = kwarg.split('=')
        args.__dict__[key] = eval(value)
        print(f'Overriding {key} to {value}')

    args.shuffle = not args.dont_shuffle
    args.verbose = not args.not_verbose
    if args.checkpoint == '':
        args.checkpoint = paths.checkpoints.lcasr
    return args
