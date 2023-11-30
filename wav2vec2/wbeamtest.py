import numpy as np
from word_beam_search import WordBeamSearch
import torch

# with open("/store/store4/data/earnings-22/all_text.txt", 'r') as f:
#     corpus = f.read().lower()

with open("words.txt", 'r') as f:
    corpus = " ".join(f.read().split('\n')).lower()
#print(corpus.lower)

lout = torch.load("logits.pt")
logits = lout['logits'].transpose(0,1)
tokenizer = tokenizer = lout['tokenizer']
print(tokenizer.vocab)
print(list(tokenizer.vocab.keys()))
print(logits.shape)

vocab = list(tokenizer.vocab.keys())
toremoveid = []
for i in range(len(vocab)):
    if vocab[i] == '|':
        vocab[i] = ' '

vocab = vocab[4:]

vocab_str = ''.join(vocab).lower()
word_vocab_str = ''.join(vocab[1:]).lower()
print(vocab_str)
print(word_vocab_str)




# swap first indice of logits with last
logits = torch.cat([logits[:,:,4:].clone(), logits[:,:,0:1].clone()], dim=-1)




# chars = "abcdefghijklmnopqrstuvwxyz "
# wordchars = "abcdefghijklmnopqrstuvwxyz"
# mat = torch.randn(100, 1    , len(chars)+1).softmax(-1).numpy()
# wbs = WordBeamSearch(10, 'NGramsForecast', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordchars.encode('utf8'))
wbs = WordBeamSearch(5, 'Words', 0.0, corpus.encode('utf8'), vocab_str.encode('utf8'), word_vocab_str.encode('utf8'))

def decode(num_seq:str):
    return ''.join([vocab[i] for i in num_seq])
print(len(vocab_str))
print(logits.shape)
# compute label string
chunks = []
#compute in chunks of 1000
for i in range(0, logits.shape[0], 1000):
    print(i)
    chunks.extend(wbs.compute(logits[i:i+1000])[0])
label_str = chunks

print(label_str)
print(decode(label_str))
