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
from lcasr.optim import madgrad
import random
from einops import rearrange
import ctc_beam_search as beam_search
#from lcasr.decoding import ctc_beam_search as beam_search
from lming.utils import general
import lcasr
from functools import partial
from matplotlib import pyplot as plt
from torch_ema import ExponentialMovingAverage
from torch.nn import functional as F
from lcasr.utils.lm_tools import add_eos, token_lens_to_mask, mark_padding
from apex.normalization import FusedLayerNorm
from lcasr.components.batchrenorm import BatchRenorm1d

def load_beamsearch(
        path:str,
        alpha:float=0.45,
        beta:float=1.53,
        prune_less_than_val:float=3.17,
        top_am_threshold:float=-6,
    ):
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
        #beam_width = 3,
        blank_id = tokenizer.vocab_size(),
        alpha = alpha,
        beta = beta,
        debug = False,
        prune_less_than_val = prune_less_than_val,
        top_am_threshold = top_am_threshold,
        max_cache_length = 128
    )
    return BeamSearch

def replace_with_frame(spec):
    for i, s in enumerate(spec):
        random_index = random.randint(0, spec.shape[-1])
        # set all frames to random frame
        spec[i] = spec[i, :, :] * 0 + spec[i, :, random_index, None]
    return spec

def frame_shuffle(spec, time_dimension=False, freq_dimension=False): # shuffles all frames in a spectrogram
    if time_dimension: spec = spec[:, :, torch.randperm(spec.shape[-1])]
    if freq_dimension: spec = spec[:, torch.randperm(spec.shape[-2]), :]
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

def get_frame_shuffle_config_from_args(args):
    frame_shuffle_args = {k.replace('frame_shuffle_', ''):v for k,v in args.__dict__.items() if k.startswith('frame_shuffle')}
    frame_shuffle_config = {
        'time_dimension': frame_shuffle_args.get('time_dimension', False),
        'freq_dimension': frame_shuffle_args.get('freq_dimension', False),
    }
    return frame_shuffle_config

def get_lr_args_from_args(args):
    lr_args = {k.replace('optim_', ''):v for k,v in args.__dict__.items() if k.startswith('optim_')}
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


def bitfit(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, FusedLayerNorm) or isinstance(module, torch.nn.LayerNorm):
            module.bias.requires_grad = True
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.requires_grad = True
        if isinstance(module, BatchRenorm1d):
            module.bias.requires_grad = True

    return model

def AWMC(
        args,
        model:nn.Module,
        spec:torch.Tensor,
        seq_len:int,
        overlap:int,
        tokenizer,
        use_tqdm:bool=True,
        optim:optim.Optimizer=madgrad.MADGRAD,
        optimizer_state:dict=None,  
        beam_search_fn:Callable=None,
        return_params:bool=False,

    ):
    
    assert beam_search_fn is None, 'Beam search function not implemented for AWMC'
    spec_augment_config = get_specaugment_config_from_args(args)
    lr_args = get_lr_args_from_args(args)
    frame_shuffle_args = get_frame_shuffle_config_from_args(args)
    
    spec_n = spec.shape[-1]
    downsampling_factor = args.config['model']['subsampling_factor']
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']

    # create copy of model parameters that are not updated
    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach().cpu() for p in original_model_params]

    if args.__dict__.get('bitfit', False):
        model = bitfit(model)

    model.train()
    ema_leader_model = ExponentialMovingAverage(model.parameters(), decay=args.__dict__.get('ema_decay', 0.999))
    ema_leader_model.update()
    ema_anchor_model = ExponentialMovingAverage(model.parameters(), decay=1.0) # no decay
    ema_anchor_model.update()

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
    epochs = args.__dict__.get('epochs', 1)
    training_data, training_keys = prepare_chunks(spec, seq_len, overlap)
    training_keys = list(training_data.keys())
    pbar = tqdm(training_keys) if use_tqdm else training_keys

    model_outputs = {}
    model.eval()
    for i in pbar:
        label_bank = [None, None]
        for j in range(epochs):
            audio_chunk = training_data[i].clone().to(model.device)
            if j == 0:
                with ema_anchor_model.average_parameters() as anchor_params, torch.no_grad() as g:
                    out = model(audio_signal = audio_chunk)
                    pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu(), decode=True)
                    #print(f'Pseudo targets: {pseudo_targets}')
                    pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets)).unsqueeze(0).to(model.device)
                    label_bank[0] = pseudo_targets.transpose(0, 1) 
                    
            with ema_leader_model.average_parameters() as leader_params, torch.no_grad() as g:
                out = model(audio_signal = audio_chunk)
                pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu(), decode=True)
                #print(f'Pseudo targets: {pseudo_targets}')
                pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets)).unsqueeze(0).to(model.device)
              
                label_bank[1] = pseudo_targets.transpose(0, 1)
              
            audio_chunk = augmentation(audio_chunk)
            audio_chunk = frame_shuffle(audio_chunk, **frame_shuffle_args)

            out = model(audio_signal = audio_chunk)
            predictions = decoder(out['final_posteriors'][-1].detach().cpu(), decode=True)
            print(f'Noisy Predictions: {predictions}')
            predictions = torch.LongTensor(tokenizer.encode(predictions)).unsqueeze(0).to(model.device)
            
            labels = [el for el in label_bank if el.shape[0] > 0]
   
            label_bank_lengths = torch.LongTensor([el.shape[0] for el in labels]).to(model.device)

            if len(labels) == 0:
                labels = [torch.LongTensor([[]]).T.to(model.device)]
                label_bank_lengths = torch.LongTensor([0]).to(model.device)
      
            labels = torch.nn.utils.rnn.pad_sequence(sequences=labels, batch_first=False, padding_value=0)
      
            labels = labels.squeeze(2).transpose(0, 1)
            N, B = out['final_posteriors'].shape[1], out['final_posteriors'].shape[0]
            total_tokens_in_loss = N * B * 2

            #print(label_bank)

            loss = ctc_loss_fn(
                out['final_posteriors'].repeat(label_bank_lengths.shape[0], 1, 1).transpose(0, 1),
                targets = labels, 
                input_lengths = torch.LongTensor([N] * labels.shape[0]).to(model.device),
                target_lengths = label_bank_lengths,
            ) / total_tokens_in_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            ema_leader_model.update()
            #ema_anchor_model.update()
            if j ==  epochs - 1:
                audio_chunk = training_data[i].clone().to(model.device)
                with torch.no_grad(): out = model(audio_signal = audio_chunk)
                logits = out['final_posteriors'][0].detach().cpu()
                logits = torch.exp(logits) # convert to prob
                
                ds_len = logits.shape[-2]
                ratio = audio_chunk.shape[-1] / ds_len
                overlap_ds = int(overlap / ratio)
          
                model_outputs[i] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}

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
    
    
    if return_params:
        updated_model_params = list(model.parameters())
        updated_model_params = [p.clone().detach().cpu() for p in updated_model_params]

    # reset model parameters
    for p, p_orig in zip(model.parameters(), original_model_params):
        p.data = p_orig.data.to(p.device)

    return logits.squeeze(0).numpy() if not return_params else (logits.squeeze(0).numpy(), updated_model_params)           
                    
            

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
        return_params:bool=False,
    ):

    spec_augment_config = get_specaugment_config_from_args(args)
    lr_args = get_lr_args_from_args(args)
    frame_shuffle_args = get_frame_shuffle_config_from_args(args)
    online_mode = args.__dict__.get('online_mode', False)
    print(spec_augment_config, lr_args, frame_shuffle_args)
    num_negatives = args.__dict__.get('num_negatives', 1)
    
    spec_n = spec.shape[-1]
    downsampling_factor = args.config['model']['subsampling_factor']
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']

    # create copy of model parameters that are not updated
    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach().cpu() for p in original_model_params]
 
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

    epochs = args.__dict__.get('epochs', 1)
    shuffle = args.__dict__.get('shuffle', False)
    online = args.__dict__.get('online', False)
    beams = args.__dict__.get('lm_tta_beams', 3)
    epochs = 1 if online else epochs
    shuffle = False if online else shuffle
    model_outputs = {}


    entropy = []
    model.eval() # don't update batchrenorm
    training_data, training_keys = prepare_chunks(spec, seq_len, overlap)
    for epoch in range(args.__dict__.get('epochs', 1)):
        print(f'Epoch {epoch + 1} / {epochs}')
        training_keys = list(training_data.keys())
        training_keys = random.sample(training_keys, len(training_keys)) if shuffle else training_keys
      
        pbar = tqdm(training_keys) if use_tqdm else training_keys
        for i in pbar:
            audio_chunk = training_data[i].clone()
            audio_chunk = audio_chunk.repeat(num_negatives+1, 1, 1) # [B, C, T]
            audio_chunk[:num_negatives] = augmentation(audio_chunk[:num_negatives]) # apply augmentation to 2 of the 3 copies
            audio_chunk[:num_negatives] = frame_shuffle(audio_chunk[:num_negatives], **frame_shuffle_args)

            u_len = audio_chunk.shape[-1]
            audio_chunk = audio_chunk.to(model.device)
            out = model(audio_signal = audio_chunk)
            # #entrop = torch.distributions.Categorical(probs = out['final_posteriors'][-1].detach().cpu().exp().mean()).entropy()
            # entrop = out['final_posteriors'][-1].detach().cpu().exp().max(-1).values
            # print(f'Entropy: {entrop.mean()}')
            # entropy.append(entrop.mean().item())
            # plt.plot(entropy)
            # plt.savefig('entropy.png')

            if beam_search_fn is None or beams == 0: 
                pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu())
            else:
                beam_search = beam_search_fn(log_probs = out['final_posteriors'][-1].detach().cpu(), beam_width = beams)
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

            if online:
                logits = out['final_posteriors'][-1].detach().cpu() 
                logits = torch.exp(logits) # convert to prob
                ds_len = logits.shape[-2]
                ratio = u_len / ds_len
                overlap_ds = int(overlap / ratio)
                model_outputs[i] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}


        
    if not online:
        model.eval()
        training_data, training_keys = prepare_chunks(spec, seq_len, overlap)
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

    if return_params:
        updated_model_params = list(model.parameters())
        updated_model_params = [p.clone().detach().cpu() for p in updated_model_params]

    # reset model parameters
    for p, p_orig in zip(model.parameters(), original_model_params):
        p.data = p_orig.data.to(p.device)


    return logits.squeeze(0).numpy() if not return_params else (logits.squeeze(0).numpy(), updated_model_params)


dynamic_eval = dynamic_eval_ctc_loss


def enc_dec_inference(
        model:nn.Module,
        spec:int,
        seq_len:int,
        overlap:int,
        tokenizer,
        use_tqdm:bool=True,
    ):
    assert overlap == 0, 'Overlap not implemented for encoder-decoder model (yet)'
    training_data, training_keys = prepare_chunks(spec, seq_len, overlap)
    training_keys_idx = [i for i in range(len(training_keys))]
    output_sequences = [None] * len(training_keys)
    pbar = tqdm(training_keys_idx) if use_tqdm else training_keys_idx
 
    for idx in pbar:
        audio_chunk = training_data[training_keys[idx]].to(model.device)
        with torch.no_grad(): output = model.generate(audio_chunk)
        text = tokenizer.decode(output.squeeze().tolist()).strip()
        print(f'Generated text: {text}')
        output_sequences[idx] = text

    transcript = " ".join(output_sequences).replace('  ', ' ').strip()
    return transcript

def enc_dec_ctc_beamsearch_inference(
        model:nn.Module,
        spec:int,
        seq_len:int,
        overlap:int,
        tokenizer,
        alpha:float = 0.45,
        beta:float = 1.53,
        prune_less_than_val:float=3.17,
        use_tqdm:bool=True,
        beam_width:int=10,
    ):
    assert overlap == 0, 'Overlap not implemented for encoder-decoder model (yet)'
    training_data, training_keys = prepare_chunks(spec, seq_len, overlap)
    training_keys_idx = [i for i in range(len(training_keys))]
    output_sequences = [None] * len(training_keys)
    pbar = tqdm(training_keys_idx) if use_tqdm else training_keys_idx
 
    for idx in pbar:
        audio_chunk = training_data[training_keys[idx]].to(model.device)
        with torch.no_grad(): text = model.ctc_beam_search(
            audio_signal = audio_chunk,
            tokenizer = tokenizer,
            alpha = alpha,
            beta = beta,
            prune_less_than_val = prune_less_than_val,
            beam_width = beam_width,
        ).strip()
            
        print(f'Generated text: {text}')
        output_sequences[idx] = text

    transcript = " ".join(output_sequences).replace('  ', ' ').strip()
    return transcript


@torch.no_grad()
def generate_enc_dec(
        model, 
        audio_signal, 
        max_generate=256, 
        bos_id=0, 
        eos_id=0,
        sample=1,
        greedy=True,
        temperature=1.0,
    ):
    '''
    sample: use multinomial sampling
    '''
    encoder_out = model.forward(audio_signal=audio_signal)
    a_hidden, length = encoder_out['a_hidden'], encoder_out['length']
    a_hidden = a_hidden.repeat(sample, 1, 1)
    text_sequence = torch.LongTensor([[bos_id]]).to(a_hidden.device).repeat(sample, 1)
    finised_sequences = []
    finished = False
    #generated = 0
    while not finished:
        decoder_logits = model.language_model_decoder(
            tokens = text_sequence,
            a_hidden = a_hidden,
            a_lengths = length,
        )['logits']
        
        probs = (decoder_logits[:, -1, :] * temperature).softmax(dim=-1)
        if sample == 1 and greedy:
            decoder_pred = probs.argmax(dim=-1)[None]
        else:
            decoder_pred = torch.multinomial(probs, num_samples=1)
        
        indices_to_drop = 0
        new_text_sequences = []
        for i in range(sample):
            if decoder_pred[i] == eos_id or text_sequence[i].shape[0] > max_generate:
                finised_sequences.append(text_sequence[i])
                indices_to_drop += 1
            else:
                new_text_sequences.append(torch.cat([text_sequence[i, None], decoder_pred[i, None]], dim=1))
        if indices_to_drop > 0: a_hidden = a_hidden[:-indices_to_drop]
        if len(new_text_sequences) > 0: text_sequence = torch.cat(new_text_sequences, dim=0)
        sample = a_hidden.shape[0]
        if sample == 0: finished = True

    
    text_lengths = torch.LongTensor([el.shape[0] for el in finised_sequences])
    text_sequence = torch.nn.utils.rnn.pad_sequence(finised_sequences, batch_first=True, padding_value=0)
        
    text_sequence = text_sequence[:, 1:] # remove bos
    text_lengths -= 1
    
    return text_sequence, encoder_out, text_lengths

def calc_loss_enc_dec(
        model,
        audio_signal,
        text_sequence,
        a_lengths,
        t_lengths,
        tokenizer,
        token_swap_prob=0.2,
        bos_id=0, 
        eos_id=0,
        label_smoothing=0.0,
    ):
    # add bos to text sequence
    #print(text_sequence.shape)
    text_sequence_bos = F.pad(text_sequence, (1, 0), value=bos_id)
    target_lengths_bos = t_lengths + 1

    targets = text_sequence_bos.clone()
    targets[:, :-1] = text_sequence_bos[:, 1:].clone()
    
    # if token_swap_prob > 0.0:
    #     swap_mask = (torch.rand(text_sequence_bos.shape) < token_swap_prob).to(text_sequence_bos.device)
    #     swap_mask[:, 0] = False # don't swap bos
    #     text_sequence_bos[swap_mask] = 

    out = model.forward(audio_signal, text_sequence_bos, a_lengths)
    ctc_out, lm_out, a_length_out = out['final_posteriors_ctc'], out['final_posteriors_lm'], out['length']

    if model.ctc_loss_weight > 0.0:
        #print(ctc_out.shape, a_length_out.shape, text_sequence.shape, t_lengths.shape)
        ctc_out = rearrange(ctc_out.repeat(t_lengths.shape[0], 1, 1), 'b n c -> n b c')
        a_length_out = a_length_out.repeat(t_lengths.shape[0])
        # print(ctc_out.shape, a_length_out.shape, text_sequence.shape, t_lengths.shape)
        # print(t_lengths.max())
        # print(text_sequence)
        ctc_loss = F.ctc_loss(
            log_probs = ctc_out,
            targets = text_sequence,
            input_lengths = a_length_out,
            target_lengths = t_lengths,
            reduction = 'sum',
            blank = ctc_out.shape[-1] - 1
        )
        a_sum = a_lengths.sum()
        ctc_loss_to_show = (ctc_loss / a_sum).item() * 100
        ctc_loss_to_bwd = ctc_loss / (ctc_out.shape[1] * ctc_out.shape[0]) * 100
    else:
        ctc_loss_to_show, ctc_loss_to_bwd = 0, 0



    if target_lengths_bos.max() == target_lengths_bos.min(): targets[:, -1] = 0
    else:
        targets = add_eos(targets, eos_id = eos_id, token_lens = target_lengths_bos)

    mask = token_lens_to_mask(target_lengths_bos)
    targets = mark_padding(targets, mask, pad_id = -100)
    predictions = lm_out
   
    #pred = predictions.squeeze(0)[:-1] # remove eos
  
    #ce = -torch.sum(teacher_probs * F.log_softmax(predictions, dim=-1), dim=-1)
    #lm_loss = torch.sum(ce * mask) / mask.sum()
    #print(lm_loss)


    lm_loss = F.cross_entropy(
        input = rearrange(predictions, 'b n c -> (b n) c'),
        target = rearrange(targets, 'b n -> (b n)'),
        ignore_index = -100,
        reduction = 'sum',
        label_smoothing = label_smoothing
    )

    lm_loss_to_show = (lm_loss / t_lengths.sum()).item()
    lm_loss_to_bwd = lm_loss / (predictions.shape[0] * predictions.shape[1])

    loss_to_show = ctc_loss_to_show * model.ctc_loss_weight + lm_loss_to_show * (1 - model.ctc_loss_weight)
    #print(1-self.ctc_loss_weight)
    loss = ctc_loss_to_bwd * model.ctc_loss_weight + lm_loss_to_bwd * (1 - model.ctc_loss_weight) 

    wandb_log_data = {
        'loss': loss_to_show,
        'ctc_loss': ctc_loss_to_show,
        'lm_loss': lm_loss_to_show,
    }

    return {
        'loss': loss,
        'display_losses': wandb_log_data,
        'ctc_posteriors': ctc_out,
        'lm_posteriors': lm_out,
        'length': a_length_out,
    }
        
def get_ema_from_args(args):
    ema_args = {k.replace('ema_', ''):v for k,v in args.__dict__.items() if k.startswith('ema_')}
    ema = ema_args.get('ema', 0.9)
    print(f'ema: {ema}')
    return ema  

def enc_dec_dynamic_eval(
        args,
        model:nn.Module,
        spec:torch.Tensor,
        seq_len:int,
        overlap:int,
        tokenizer,
        use_tqdm:bool=True,
        optim:optim.Optimizer=madgrad.MADGRAD,
        optimizer_state:dict=None,
        beam_search_fn:Callable=None,
        return_params:bool=False,
    ):

    spec_augment_config = get_specaugment_config_from_args(args)
    lr_args = get_lr_args_from_args(args)
    #print(spec_augment_config, lr_args)
    num_negatives = 1 # only coded for 1 negative

    ###
    spec_n = spec.shape[-1]
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']

    # create copy of model parameters that are not updated
    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach().cpu() for p in original_model_params]
 
    # ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='sum')
    # ce_loss_fn = torch.nn.CrossEntropyLoss()
    
    # freeze params
    modules_to_freeze = [
        model.language_model_decoder.pos_enc,
        model.pos_enc,
    ]

    dropout_emb = args.__dict__.get('dropout_emb', 0.0)
    dropout_post_ff = args.__dict__.get('dropout_post_ff', 0.0) 
    dropout_attn = args.__dict__.get('dropout_attn', 0.0)


    model.language_model_decoder.dropout_emb = dropout_emb
    model.language_model_decoder.ff_out_dropout = dropout_post_ff
        # layer[0].fn.flash_attn_fn.dropout_p = dropout_attn
        # layer[0].fn.flash_attn_c_fn.drop.p = dropout_attn
    for layer in model.language_model_decoder.layers:
        layer[0].fn.dropout_p = 0

    #model.language_model_decoder

    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False
        print(f'Freezing {module}')

    ema = ExponentialMovingAverage(model.parameters(), decay=get_ema_from_args(args))

    optimizer = optim(model.parameters(), **lr_args)

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        
        
    augmentation = SpecAugment(**spec_augment_config)

    if seq_len > spec_n:
        seq_len, overlap = spec_n, 0
    else:
        overlap = overlap if overlap != -1 else args.config['audio_chunking']['overlap']

    assert overlap == 0, 'Overlap > 0 not implemented for encoder-decoder model'
    print(f'Using seq_len: {seq_len}')# and overlap: {overlap}')


    
    #model.ctc_loss_weight = 1.0
    ema.update()
    model.eval() # don't update batchrenorm
    training_data, training_keys = prepare_chunks(spec, seq_len, overlap)
    for epoch in range(args.__dict__.get('epochs', 1)):
        print(f'Epoch {epoch + 1} / {args.__dict__.get("epochs", 1)}')
        training_keys = list(training_data.keys())
        training_keys_idx = [i for i in range(len(training_keys))]
        training_keys_idx = random.sample(training_keys_idx, len(training_keys_idx)) if args.__dict__.get('shuffle', False) else training_keys_idx

        pbar = tqdm(training_keys_idx) if use_tqdm else training_keys_idx
        for idx in pbar:
            audio_chunk = training_data[training_keys[idx]].clone().to(model.device)
            audio_chunk = audio_chunk.repeat(num_negatives+1, 1, 1)
            audio_chunk[:num_negatives] = augmentation(audio_chunk[:num_negatives]) # apply augmentation to 2 of the 3 copies

            with ema.average_parameters():
                teacher_pred, teacher_encoder_out, text_lengths = generate_enc_dec(
                    model, 
                    audio_chunk[-1, None],
                    max_generate = 256,
                    sample = args.__dict__.get('enc_dec_sample', 1),
                    greedy = args.__dict__.get('enc_dec_greedy', True),
                    temperature = args.__dict__.get('enc_dec_temperature', 1.0),
                )

            teacher_pred_text = tokenizer.decode(teacher_pred[0].tolist()).strip()
            print(f'Teacher pred: {teacher_pred_text}')
            teacher_lengths = text_lengths.to(model.device)
            acoustic_length = torch.LongTensor([audio_chunk.shape[-1]]).to(model.device)
            #print(audio_chunk[:num_negatives].shape, teacher_pred.shape)

            for layer in model.language_model_decoder.layers:
                layer[0].fn.dropout_p = dropout_attn
                
            model.language_model_decoder.train() # for dropout
            student_out = calc_loss_enc_dec(
                model = model,
                audio_signal = audio_chunk[:num_negatives], 
                text_sequence = teacher_pred, 
                a_lengths = acoustic_length, 
                t_lengths = teacher_lengths,
                tokenizer = tokenizer
            )
            model.language_model_decoder.eval()

            for layer in model.language_model_decoder.layers:
                layer[0].fn.dropout_p = 0



            loss = student_out['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    final_out = enc_dec_inference(
        model = model,
        spec = spec,
        seq_len = seq_len,
        overlap = overlap,
        tokenizer = tokenizer,
        use_tqdm = use_tqdm
    )
    if return_params:
        updated_model_params = list(model.parameters())
        updated_model_params = [p.clone().detach().cpu() for p in updated_model_params]

    # reset model parameters
    for p, p_orig in zip(model.parameters(), original_model_params):
        p.data = p_orig.data.to(p.device)

    return final_out if not return_params else (final_out, updated_model_params)




# def dynamic_eval_enc_dec_loss(
#         args, 
#         model:nn.Module, 
#         spec:torch.Tensor, 
#         seq_len:int, 
#         overlap:int, 
#         tokenizer, 
#         use_tqdm=True,
#         optim:optim.Optimizer=madgrad.MADGRAD,
#         optimizer_state:dict=None,
#         beam_search_fn:Callable=None,
#         return_params:bool=False,
#     ):

#     assert beam_search_fn is None, 'Beam search not implemented for encoder-decoder model ()

'''
shared functions between scripts
'''
def apply_args(parser):
    parser.add_argument('-c', '--checkpoint', type=str, default='', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=16384, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-o', '--overlap', type=int, default=14336, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-nv', '--not_verbose', action='store_true', help='verbose')
    parser.add_argument('-log', '--log', type=str, default='')
    parser.add_argument('-ds', '--dont_shuffle', action='store_true', help='dont shuffle')
    #parser.add_argument('-shuffle', '--shuffle', action='store_true', help='shuffle')
    parser.add_argument('-epochs', '--epochs', type=int, default=1, help='epochs')
    parser.add_argument('-dfa', '--disable_flash_attention', action='store_true', help='disable flash attention')
    parser.add_argument('-beamsearch', '--beamsearch', action='store_true', help='use beam search')
    parser.add_argument('-kwargs', '--kwargs', nargs='+', help='kwargs')
    parser.add_argument('-awmc', '--awmc', action='store_true', help='Use AWMC method from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10389640&tag=1 instead of dynamic eval')

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