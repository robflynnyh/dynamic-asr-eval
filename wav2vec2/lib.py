from omegaconf import OmegaConf
import os
paths = OmegaConf.load(os.path.join(os.path.dirname(__file__), '../paths.yaml'))
import torchaudio, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F, madgrad
from lcasr.utils.augmentation import SpecAugment
from lcasr.decoding.greedy import GreedyCTCDecoder
import random
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCTC
import numpy as np
import augment # https://github.com/facebookresearch/WavAugment
import matplotlib.pyplot as plt
from soft_dtw_cuda import SoftDTW

from torch_ema import ExponentialMovingAverage


def load_pretrained_model(name = "facebook/wav2vec2-base-960h"):
    asr_model = AutoModelForCTC.from_pretrained(name)
    tokenizer = AutoProcessor.from_pretrained(name)
    return asr_model, tokenizer

def load_audio(path):
    audio_signal, sampling_rate = torchaudio.load(path)
    return audio_signal, sampling_rate


def preprocess(audio_path):
    audio_signal, sr = load_audio(audio_path)
    return audio_signal

def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0
    return model
            

def dynamic_eval_ctc_loss(
        args, 
        model:nn.Module, 
        spec:torch.Tensor, 
        seq_len:int, 
        overlap:int, 
        tokenizer, 
        processor,
        use_tqdm=True,
        optim:optim.Optimizer=madgrad.MADGRAD,
        num_negatives:int=1,
        lr_args:dict={'lr':1e-7},

    ):
    # torch.use_deterministic_algorithms(True) # so conv is deterministic
    # torch.backends.cudnn.deterministic = True

    spec_n = spec.shape[-1]
    downsampling_factor = 4
    seq_len = seq_len 

    # create copy of model parameters that are not updated
    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach() for p in original_model_params]

    ctc_loss_fn = torch.nn.CTCLoss(blank=tokenizer.blank_id, reduction='sum')
    print(tokenizer.vocab)

    # print(model)
    #exit()

    # freeze all in model.wav2vec2.feature_projection and model.wav2vec2.feature_extractor
    # for param in model.wav2vec2.feature_projection.parameters():
    #     param.requires_grad = False
    # for param in model.wav2vec2.feature_extractor.parameters():
    #     param.requires_grad = False

    # # add forward hook on each layer
    # def hook_fn_forward(module, input, output):
 
    #     h = output[0]
    #     h_aug = h[:-1]
    #     h_dims = h.shape[-1]

    #     a = torch.fft.rfft(h_aug, dim=-1)
    #     # mask 30% of the frequencies
    #     mask = torch.rand(a.shape[-1]) < 0.1
    #     mask = mask.to(a.device)
    #     a = a.masked_fill(mask, a.mean())
    #     # revert to time domain
    #     cz = torch.fft.irfft(a, h_aug.shape[-1], dim=-1)
  

    #     #print(f'Forward hook: {output[0].shape}')
    #     return [torch.cat((cz, h[-1:]), dim=0), *output[1:]]
    # for i, layer in enumerate(model.wav2vec2.encoder.layers):
    #     layer.register_forward_hook(hook_fn_forward)


    optimizer = optim(model.parameters(), **lr_args)
    #ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = tokenizer.blank_id)


        
    if seq_len > spec_n:
        seq_len, overlap = spec_n, 0
   
    assert overlap / downsampling_factor == overlap // downsampling_factor, 'Overlap must be a multiple of the downsampling factor'
    print(f'Using seq_len: {seq_len} and overlap: {overlap}')

    all_logits, logit_count = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size )), torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size ))
    last_ulen, kill_next, logit_position = None, False, 0

    losses = []
    training_data = {}
    for i in range(0, spec_n, seq_len-overlap):
        #print(f'{i} - {i+seq_len}')
        audio_chunk = spec[:, i:i+seq_len] # [B, C, T]
        u_len = audio_chunk.shape[-1]
        if kill_next:
            break
        elif last_ulen != None and u_len < last_ulen:
            kill_next = True
        last_ulen = u_len
        training_data[i] = audio_chunk

    print(f'Number of training examples: {len(training_data)}')

    soft_dtw = SoftDTW(use_cuda=True, gamma=1.5)
    model = disable_dropout(model)
    for epoch in range(args.__dict__.get('epochs', 1)):
        print(f'Epoch {epoch + 1} / {args.__dict__.get("epochs", 1)}')
        model_outputs = {}
        training_keys = list(training_data.keys())
        training_keys = random.sample(training_keys, len(training_keys)) if args.__dict__.get('shuffle', False) else training_keys

        pbar = tqdm(training_keys) if use_tqdm else training_keys
        for i in pbar:
            audio_chunk = training_data[i].clone()
            
            audio_chunk = audio_chunk.repeat(num_negatives+2, 1, 1).contiguous() # [B, C, T]


            noise_generator = lambda: torch.zeros((1, audio_chunk.shape[-1]))
            augmentation_1 = augment.EffectChain().time_dropout(max_seconds=0.1)
            augmentation_2 = augment.EffectChain().additive_noise(noise_generator, snr=0).reverb(50, 50, 100)

            src_info = {'rate': 16000, 'channels': 1, 'mean':audio_chunk[-1].mean().item()}

            tgt_info = {'rate': 16000, 'channels': 1}

            for j in range(num_negatives):
                for _ in range(100):
                    audio_chunk[j] = augmentation_1.apply(audio_chunk[j], src_info, tgt_info).clone()
                audio_chunk[j] = augmentation_2.apply(audio_chunk[j], src_info, tgt_info).clone()[-1,None]
       

            u_len = audio_chunk.shape[-1]
            audio_chunk = audio_chunk.squeeze(1)
            input_values = processor.feature_extractor(audio_chunk.numpy(), sampling_rate=16000, return_tensors='pt').input_values.to(model.device)
           
            pred = model(input_values[:-1])
            # with ema.average_parameters():
            #     tgt = model(input_values[-1:])

            logits = pred.logits
            print(logits.shape)
            log_p = F.log_softmax(logits, dim=-1)
            pseudo_targets = decoder(log_p[-1].detach().cpu())
            dtgts = decoder(log_p[0].detach().cpu())
            print(pseudo_targets, '\n', dtgts, '\n---\n')
             
            pseudo_targets = torch.LongTensor(tokenizer(pseudo_targets).input_ids).unsqueeze(0).to(model.device).repeat(num_negatives, 1)
            augmented_outs = log_p[:num_negatives]            
            
            N, B = augmented_outs.shape[1], augmented_outs.shape[0]
            total_tokens_in_loss = N * B
  
            loss = ctc_loss_fn(augmented_outs.transpose(0, 1), pseudo_targets, torch.LongTensor([N] * augmented_outs.shape[0]).to(model.device), torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0]).to(model.device)) / total_tokens_in_loss
            print('loss', loss.item())
            losses.append(loss.item())

            # pseudo_targets = logits[-1].detach().unsqueeze(0)
            # psuedo_text = decoder(log_p[-1].detach().cpu())
            # print(psuedo_text)
            # predictions = logits[:-1]
            # pseudo_targets = pseudo_targets.repeat(predictions.shape[0], 1, 1)
            # loss = soft_dtw(pseudo_targets, predictions).mean()
            # print('loss', loss.item())
            # losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            #clip gradients
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            #ema.update()

            plt.plot(losses)
            plt.savefig('loss.png')
            plt.close()
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
    # save logits

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
    parser.add_argument('-seq', '--seq_len', type=int, default=131072)
    parser.add_argument('-overlap', '--overlap', type=int, default=0)
    parser.add_argument('-nv', '--not_verbose', action='store_true', help='verbose')
    parser.add_argument('-log', '--log', type=str, default='')
    parser.add_argument('-shuffle', '--shuffle', action='store_true', help='shuffle')
    parser.add_argument('-epochs', '--epochs', type=int, default=1, help='epochs')
    parser.add_argument('-dfa', '--disable_flash_attention', action='store_true', help='disable flash attention')

    args = parser.parse_args()
    args.verbose = not args.not_verbose
    if args.checkpoint == '':
        args.checkpoint = paths.checkpoints.wav2vec2
    return args