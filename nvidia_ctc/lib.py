from omegaconf import OmegaConf
import os
paths = OmegaConf.load(os.path.join(os.path.dirname(__file__), '../paths.yaml'))
import nemo.collections.asr as nemo_asr
import torchaudio, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F, madgrad
from lcasr.utils.augmentation import SpecAugment
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.components.batchrenorm import BatchRenorm1d
import random
from tqdm import tqdm

def load_pretrained_model(name):
    asr_model = nemo_asr.models.ASRModel.from_pretrained(name)
    return asr_model

def load_audio(path):
    audio_signal, sampling_rate = torchaudio.load(path)
    return audio_signal, sampling_rate

def get_spectrogram(audio_signal, model, device):
    a_length = torch.LongTensor([audio_signal.shape[1]])
    spec, _ = model.preprocessor.to(device)(input_signal = audio_signal.to(device), length = a_length.to(device))
    return spec

def preprocess(audio_path, model):
    audio_signal, sr = load_audio(audio_path)
    spec = get_spectrogram(audio_signal, model, 'cpu')
    return spec

def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0

def dynamic_eval_ctc_loss(
        args, 
        model:nn.Module, 
        spec:torch.Tensor, 
        seq_len:int, 
        overlap:int, 
        tokenizer, 
        use_tqdm=True,
        optim:optim.Optimizer=optim.Adam,
        num_negatives:int=4,
        lr_args:dict={'lr':1e-8},
        spec_augment_config={
            'n_time_masks': 3,
            'n_freq_masks': 3,
            'freq_mask_param': 40,
            'time_mask_param': -1,
            'min_p': 0.05,
            'zero_masking': False,
        }
    ):
    model.spec_augmentation = None
    disable_dropout(model)

    spec_n = spec.shape[-1]
    downsampling_factor = 8
    seq_len = seq_len 

    # create copy of model parameters that are not updated
    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach() for p in original_model_params]

    ctc_loss_fn = torch.nn.CTCLoss(blank=tokenizer.vocab_size, reduction='sum')

    # print(model)
    #exit()
    
    decoder = GreedyCTCDecoder(tokenizer = SentencePieceAdapter(tokenizer), blank_id = tokenizer.vocab_size)
    augmentation = SpecAugment(**spec_augment_config)

    model.train()

    # model.eval()
    # model.encoder.pre_encode.eval()
    # model.encoder.pos_enc.eval()
    # model.decoder.eval()
    #freeze above layers
    for param in model.encoder.pre_encode.parameters():
        param.requires_grad = False
    for param in model.encoder.pos_enc.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False
        
    # print(model)
    for layer in model.encoder.layers:
        # for param in layer.conv.parameters():
        #     param.requires_grad = False
        layer.conv.eval()
        layer.conv.batch_norm.eval()
        bn = layer.conv.batch_norm
        shape = bn.weight.shape[-1]
        brn = BatchRenorm1d(shape, momentum=0.001, eps=1e-5)
        brn.num_batches_tracked = torch.tensor(1000000, dtype=torch.long)
        brn.weight.data = bn.weight.data
        brn.bias.data = bn.bias.data
        brn.running_mean.data = bn.running_mean.data
        brn.running_std.data = bn.running_var.data.sqrt()
        layer.conv.batch_norm = brn.to(bn.weight.device)
    #model.encoder.layers[0].conv.batch_norm.eval()

    optimizer = optim(model.parameters(), **lr_args)

    if seq_len > spec_n:
        seq_len, overlap = spec_n, 0
   
    assert overlap / downsampling_factor == overlap // downsampling_factor, 'Overlap must be a multiple of the downsampling factor'
    print(f'Using seq_len: {seq_len} and overlap: {overlap}')

    all_logits, logit_count = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size + 1)), torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size + 1))
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

            u_len = audio_chunk.shape[-1]
            audio_chunk = audio_chunk.to(model.device)
            out = model(processed_signal=audio_chunk, processed_signal_length=torch.LongTensor([u_len] * audio_chunk.shape[0]).to(model.device))
            log_p, e_lens, greedy_pred = out
            print(log_p.shape, e_lens.shape, greedy_pred.shape)
            pseudo_targets = decoder(log_p[-1].detach().cpu())
            print(pseudo_targets)
            pseudo_targets = torch.LongTensor(tokenizer.text_to_ids(pseudo_targets)).unsqueeze(0).to(model.device).repeat(num_negatives, 1)
            augmented_outs = log_p[:num_negatives]            
            
            N, B = augmented_outs.shape[1], augmented_outs.shape[0]
            total_tokens_in_loss = N * B
  
            loss = ctc_loss_fn(augmented_outs.transpose(0, 1), pseudo_targets, torch.LongTensor([N] * augmented_outs.shape[0]).to(model.device), torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0]).to(model.device)) / total_tokens_in_loss


            optimizer.zero_grad()
            loss.backward()
            # clip grad norm to 1.0
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            
            #print(model.encoder.layers[0].conv.batch_norm.running_mean) # check that batch norm is frozen
        
            logits = log_p[-1].detach().cpu()
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

class SentencePieceAdapter():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def decode(self, ids):
        return self.tokenizer.ids_to_text(ids)

    def encode(self, text):
        return self.tokenizer.text_to_ids(text)

def apply_args(parser):
    parser.add_argument('-c', '--checkpoint', type=str, default='', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=2048)
    parser.add_argument('-overlap', '--overlap', type=int, default=0)
    parser.add_argument('-nv', '--not_verbose', action='store_true', help='verbose')
    parser.add_argument('-log', '--log', type=str, default='')
    parser.add_argument('-shuffle', '--shuffle', action='store_true', help='shuffle')
    parser.add_argument('-epochs', '--epochs', type=int, default=1, help='epochs')
    parser.add_argument('-dfa', '--disable_flash_attention', action='store_true', help='disable flash attention')

    args = parser.parse_args()
    args.verbose = not args.not_verbose
    if args.checkpoint == '':
        args.checkpoint = paths.checkpoints.nvidia_ctc
    return args