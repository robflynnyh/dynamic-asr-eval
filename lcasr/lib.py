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
from lcasr.decoding import ctc_beam_search as beam_search
from lming.utils import general
import lcasr
from functools import partial
from matplotlib import pyplot as plt
from torch_ema import ExponentialMovingAverage
from torch.nn import functional as F
from lcasr.utils.lm_tools import add_eos, token_lens_to_mask, mark_padding

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
    
    assert beam_search_fn is not None, 'Beam search function must be provided for AWMC'
    spec_augment_config = get_specaugment_config_from_args(args)
    lr_args = get_lr_args_from_args(args)
    frame_shuffle_args = get_frame_shuffle_config_from_args(args)
    
    spec_n = spec.shape[-1]
    downsampling_factor = args.config['model']['subsampling_factor']
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']

    # create copy of model parameters that are not updated
    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach().cpu() for p in original_model_params]

    model.train()
    ema_leader_model = ExponentialMovingAverage(model.parameters(), decay=0.999)
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
    
    training_data, training_keys = prepare_chunks(spec, seq_len, overlap)
    training_keys = list(training_data.keys())
    pbar = tqdm(training_keys) if use_tqdm else training_keys
    for i in pbar:
        label_bank = [None, None]
        for j in range(args.__dict__.get('epochs', 1)):
            audio_chunk = training_data[i].clone()
            if j == 0:
                with ema_anchor_model.average_parameters() as anchor_params, torch.no_grad() as g:
                    out = model(audio_signal = audio_chunk)
                    pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu(), decode=False)
                    label_bank[0] = torch.LongTensor(pseudo_targets).unsqueeze(0).to(model.device)

            with ema_leader_model.average_parameters() as leader_params, torch.no_grad() as g:
                out = model(audio_signal = audio_chunk)
                pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu(), decode=True)
                pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets)).unsqueeze(0).to(model.device)
                print(f'Pseudo targets: {pseudo_targets}')
                label_bank[1] = pseudo_targets

            audio_chunk = augmentation(audio_chunk)
            audio_chunk = frame_shuffle(audio_chunk, **frame_shuffle_args)
            out = model(audio_signal = audio_chunk)
            predictions = decoder(out['final_posteriors'][-1].detach().cpu(), decode=True)
            print(f'Noisy Predictions: {predictions}')
            predictions = torch.LongTensor(tokenizer.encode(predictions)).unsqueeze(0).to(model.device)

            # pad label bank to same length
            print(label_bank[0].shape, label_bank[1].shape)
            label_bank_lengths = torch.LongTensor([label_bank[0].shape[-1], label_bank[1].shape[-1]])
            max_length = label_bank_lengths.max()
            label_bank = torch.nn.utils.rnn.pad_sequence(label_bank, batch_first=True, padding_value=-100)
            print(label_bank.shape)
            print(label_bank)
            exit()           

            
                    
            

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
            audio_chunk[:num_negatives] = frame_shuffle(audio_chunk[:num_negatives], **frame_shuffle_args)

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


@torch.no_grad()
def generate_enc_dec(model, audio_signal, max_generate=256, bos_id=0, eos_id=0):
    '''
    greedy generation, audio_signal should be a single batch
    '''
    encoder_out = model.forward(audio_signal=audio_signal)
    a_hidden, length = encoder_out['a_hidden'], encoder_out['length']
    text_sequence = torch.LongTensor([[bos_id]]).to(a_hidden.device)
    finished = False
    #generated = 0
    output_probs = []
    while not finished:
        decoder_logits = model.language_model_decoder(
            tokens = text_sequence,
            a_hidden = a_hidden,
            a_lengths = length,
        )
        probs = decoder_logits[0, -1, :].softmax(dim=-1)
        output_probs.append(probs)
        decoder_pred = probs.argmax(dim=-1)
        
        #generated += 1
        #print(f'Generated {generated} tokens: {decoder_pred.item()}')
        if decoder_pred == eos_id or text_sequence.shape[1] > max_generate:
            finished = True
        else:
            text_sequence = torch.cat([text_sequence, decoder_pred.unsqueeze(0).unsqueeze(0)], dim=1)

    output_probs = torch.stack(output_probs, dim=0)
    text_sequence = text_sequence[:, 1:] # remove bos
    
    return text_sequence, encoder_out, output_probs

def calc_loss_enc_dec(
        model,
        audio_signal,
        text_sequence,
        teacher_probs,
        a_lengths,
        t_lengths,
        bos_id=0, 
        eos_id=0,
        label_smoothing=0.0
    ):
    # add bos to text sequence
    print(text_sequence.shape)
    text_sequence_bos = F.pad(text_sequence, (1, 0), value=bos_id)
    target_lengths_bos = t_lengths + 1
    
    out = model.forward(audio_signal, text_sequence_bos, a_lengths)
    ctc_out, lm_out, a_length_out = out['final_posteriors_ctc'], out['final_posteriors_lm'], out['length']

    if model.ctc_loss_weight > 0.0:
        ctc_loss = F.ctc_loss(
            log_probs = rearrange(ctc_out, 'b n c -> n b c'),
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

    targets = text_sequence_bos.clone()
    targets[:, :-1] = text_sequence_bos[:, 1:]

    if target_lengths_bos.max() == target_lengths_bos.min(): targets[:, -1] = 0
    else:
        targets = add_eos(targets, eos_id = eos_id, token_lens = target_lengths_bos)

    mask = token_lens_to_mask(target_lengths_bos)
    targets = mark_padding(targets, mask, pad_id = -100)
    predictions = lm_out

    #pred = predictions.squeeze(0)[:-1] # remove eos
    print(predictions.shape, teacher_probs.shape)
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
    print(ema)
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
    print(spec_augment_config, lr_args)
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
        #model.language_model_decoder.embed,
        model.pos_enc,
        #model.subsampling
        #model.language_model_decoder
    ]

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
                teacher_pred, teacher_encoder_out, output_probs = generate_enc_dec(model, audio_chunk[-1, None])

            teacher_pred_text = tokenizer.decode(teacher_pred.squeeze().tolist()).strip()
            print(f'Teacher pred: {teacher_pred_text}')
            teacher_lengths = torch.LongTensor([teacher_pred.shape[-1]]).to(model.device)
            acoustic_length = torch.LongTensor([audio_chunk.shape[-1]]).to(model.device)
            #print(audio_chunk[:num_negatives].shape, teacher_pred.shape)


            student_out = calc_loss_enc_dec(
                model = model,
                audio_signal = audio_chunk[:num_negatives], 
                text_sequence = teacher_pred, 
                teacher_probs = output_probs,
                a_lengths = acoustic_length, 
                t_lengths = teacher_lengths,
            )
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
