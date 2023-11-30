from omegaconf import OmegaConf
import os
#paths = OmegaConf.load('paths.yaml')
# use absolute path
paths = OmegaConf.load(os.path.join(os.path.dirname(__file__), '../paths.yaml'))
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from lcasr.utils.augmentation import SpecAugment
from lcasr.decoding.greedy import GreedyCTCDecoder
import madgrad, random
from soft_dtw_cuda import SoftDTW
from einops import rearrange


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
        lr_args:dict={'lr':9e-5},
        optimizer_state:dict=None,
        spec_augment_config={
            'n_time_masks': 2,
            'n_freq_masks': 3,
            'freq_mask_param': 42,
            'time_mask_param': -1,
            'min_p': 0.05,
            'zero_masking': False,
        }
    ):

    spec_n = spec.shape[-1]
    downsampling_factor = args.config['model']['subsampling_factor']
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']

    # create copy of model parameters that are not updated
    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach() for p in original_model_params]

    # for name, param in model.named_parameters():
    #     if 'layers.2' not in name:
    #         param.requires_grad = False
    #         print(f'Freezing {name}')
    #     else:
    #         print(f'Updating {name}')

    #model.self_conditioning = False

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
    last_ulen, kill_next, logit_position = None, False, 0

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


    for epoch in range(args.__dict__.get('epochs', 1)):
        print(f'Epoch {epoch + 1} / {args.__dict__.get("epochs", 1)}')
        model_outputs = {}
        training_keys = list(training_data.keys())
        training_keys = random.sample(training_keys, len(training_keys)) if args.__dict__.get('shuffle', False) else training_keys

        pbar = tqdm(training_keys) if use_tqdm else training_keys
        for i in pbar:
            audio_chunk = training_data[i].clone()
            audio_chunk = audio_chunk.repeat(num_negatives+1, 1, 1) # [B, C, T]
            audio_chunk[:num_negatives] = augmentation(audio_chunk[:num_negatives]) # apply augmentation to 2 of the 3 copies
            # for neg in range(num_negatives):
            #     audio_chunk[neg] = spec_shuffle(audio_chunk[neg])

            u_len = audio_chunk.shape[-1]
            audio_chunk = audio_chunk.to(model.device)
            out = model(audio_signal = audio_chunk)

            pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu())
            noisy_predictions = decoder(out['final_posteriors'][0].detach().cpu())
            print(f'Pseudo targets: {pseudo_targets}')
            print(f'Noisy predictions: {noisy_predictions}')
            print('\n--\n')
            pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets)).unsqueeze(0).to(model.device).repeat(num_negatives, 1)
            augmented_outs = out['final_posteriors'][:num_negatives]            
            
            N, B = augmented_outs.shape[1], augmented_outs.shape[0]
            total_tokens_in_loss = N * B
            # get total squared difference between augmented outputs
            # print(f'Augmented output difference: {augmented_outs_diff[-1]}, Augmented output CTC loss: {augmented_ctc_loss}')
            loss = ctc_loss_fn(augmented_outs.transpose(0, 1), pseudo_targets, torch.LongTensor([N] * augmented_outs.shape[0]).to(model.device), torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0]).to(model.device)) / total_tokens_in_loss
    
            # l2_pretrained_loss = 0
            # num_params = 0
            # for param, original_param in zip(model.parameters(), original_model_params):
            #     l2_pretrained_loss += torch.sum((param - original_param) ** 2)
            #     num_params += torch.numel(param)
            # loss = loss + (l2_pretrained_loss / num_params) * 1.0

            optimizer.zero_grad()
            loss.backward()
            # clip gradients
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8) # add clip value to args
            optimizer.step()
        
            logits = out['final_posteriors'][-1].detach().cpu()
            logits = torch.exp(logits) # convert to prob
            ds_len = logits.shape[-2]
            ratio = u_len / ds_len
            overlap_ds = int(overlap / ratio)
            model_outputs[i] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}
    
           
    
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


def dynamic_eval_ctc_loss_2(
        args, 
        model:nn.Module, 
        spec:torch.Tensor, 
        seq_len:int, 
        overlap:int, 
        tokenizer, 
        use_tqdm=True,
        optim:optim.Optimizer=madgrad.MADGRAD,
        num_negatives:int=2,
        lr_args:dict={'lr':9e-5},
        optimizer_state:dict=None,
        spec_augment_config={
            'n_time_masks': 2,
            'n_freq_masks': 3,
            'freq_mask_param': 42,
            'time_mask_param': -1,
            'min_p': 0.05,
            'zero_masking': False,
        }
    ):

    spec_n = spec.shape[-1]
    downsampling_factor = args.config['model']['subsampling_factor']
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']

    # create copy of model parameters that are not updated
    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach() for p in original_model_params]

    # for name, param in model.named_parameters():
    #     if 'layers.2' not in name:
    #         param.requires_grad = False
    #         print(f'Freezing {name}')
    #     else:
    #         print(f'Updating {name}')

    #model.self_conditioning = False

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
    last_ulen, kill_next, logit_position = None, False, 0

    soft_dtw = SoftDTW(use_cuda=True, gamma=2.01)
    activations = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activations[name] = output[0]
        return hook
    for i, layer in enumerate(model.layers):
        layer.register_forward_hook(getActivation(f'{i}'))

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


    for epoch in range(args.__dict__.get('epochs', 1)):
        print(f'Epoch {epoch + 1} / {args.__dict__.get("epochs", 1)}')
        model_outputs = {}
        training_keys = list(training_data.keys())
        training_keys = random.sample(training_keys, len(training_keys)) if args.__dict__.get('shuffle', False) else training_keys

        pbar = tqdm(training_keys) if use_tqdm else training_keys
        for i in pbar:
            audio_chunk = training_data[i].clone()
            audio_chunk = audio_chunk.repeat(num_negatives+1, 1, 1) # [B, C, T]
            audio_chunk[:num_negatives] = augmentation(audio_chunk[:num_negatives]) # apply augmentation to 2 of the 3 copies
            # for neg in range(num_negatives):
            #     audio_chunk[neg] = spec_shuffle(audio_chunk[neg])

            u_len = audio_chunk.shape[-1]
            audio_chunk = audio_chunk.to(model.device)
            out = model(audio_signal = audio_chunk)

            pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu())
            noisy_predictions = decoder(out['final_posteriors'][0].detach().cpu())
            print(f'Pseudo targets: {pseudo_targets}')
            print(f'Noisy predictions: {noisy_predictions}')
            print('\n--\n')
            pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets)).unsqueeze(0).to(model.device).repeat(num_negatives, 1)
            augmented_outs = out['final_posteriors'][:num_negatives]            
            
            N, B = augmented_outs.shape[1], augmented_outs.shape[0]
            total_tokens_in_loss = N * B
            # get total squared difference between augmented outputs
            # print(f'Augmented output difference: {augmented_outs_diff[-1]}, Augmented output CTC loss: {augmented_ctc_loss}')
            loss = ctc_loss_fn(augmented_outs.transpose(0, 1), pseudo_targets, torch.LongTensor([N] * augmented_outs.shape[0]).to(model.device), torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0]).to(model.device)) / total_tokens_in_loss
            act_5 = activations['5'][-1][None]
            act_4 = activations['5'][0][None]
            print(f'Activation 5: {act_5.shape}, Activation 4: {act_4.shape}')
            # check if divisible by 16
            if act_5.shape[1] % 16 == 0:
                mapper = lambda x: rearrange(x, 'b (a c) d -> (b a) c d', a=16)
                act_4, act_5 = mapper(act_4), mapper(act_5)
                dtw_loss = soft_dtw(act_5.detach(), act_4).sum() / 2048
                print(f'DTW loss: {dtw_loss}')
                loss = loss + dtw_loss*0.0001 

            # l2_pretrained_loss = 0
            # num_params = 0
            # for param, original_param in zip(model.parameters(), original_model_params):
            #     l2_pretrained_loss += torch.sum((param - original_param) ** 2)
            #     num_params += torch.numel(param)
            # loss = loss + (l2_pretrained_loss / num_params) * 1.0

            optimizer.zero_grad()
            loss.backward()
            # clip gradients
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8) # add clip value to args
            optimizer.step()
        
            logits = out['final_posteriors'][-1].detach().cpu()
            logits = torch.exp(logits) # convert to prob
            ds_len = logits.shape[-2]
            ratio = u_len / ds_len
            overlap_ds = int(overlap / ratio)
            model_outputs[i] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}
    
           
    
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

dynamic_eval = dynamic_eval_ctc_loss


'''
shared functions between scripts
'''
def apply_args(parser):
    parser.add_argument('-c', '--checkpoint', type=str, default='', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-nv', '--not_verbose', action='store_true', help='verbose')
    parser.add_argument('-log', '--log', type=str, default='')
    parser.add_argument('-shuffle', '--shuffle', action='store_true', help='shuffle')
    parser.add_argument('-epochs', '--epochs', type=int, default=1, help='epochs')
    parser.add_argument('-dfa', '--disable_flash_attention', action='store_true', help='disable flash attention')

    args = parser.parse_args()
    args.verbose = not args.not_verbose
    if args.checkpoint == '':
        args.checkpoint = paths.checkpoints.lcasr
    return args
