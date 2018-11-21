import sys
import re
import time
import torch
from model import *
from utils import *
from os.path import isfile
from torch.autograd import Variable
from gensim.models import FastText
from tqdm import tqdm

fastText = FastText.load_fasttext_format('wordvec/skipgram.bin')

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(EMBED_SIZE, 2, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def word2vec(x):
    try:
        return torch.from_numpy(fastText.wv[x])
    except:
        return torch.rand(512)

def load_data():
    data = []
    src_batch = []
    tgt_batch = []
    src_batch_len = 0
    tgt_batch_len = 0
    print("loading data...")
    src_vocab = load_vocab(sys.argv[2], "src")
    tgt_vocab = load_vocab(sys.argv[3], "tgt")
    fo = open(sys.argv[4], "r")
    for line in fo:
        line = line.strip()
        src, tgt = line.split("\t")
        src = [int(i) for i in src.split(" ")] + [EOS_IDX]
        tgt = [int(i) for i in tgt.split(" ")] + [EOS_IDX]
        if len(src) > src_batch_len:
            src_batch_len = len(src)
        if len(tgt) > tgt_batch_len:
            tgt_batch_len = len(tgt)
        src_batch.append(src)
        tgt_batch.append(tgt)
        if len(src_batch) == BATCH_SIZE:
            for seq in src_batch:
                seq.extend([PAD_IDX] * (src_batch_len - len(seq)))
            for seq in tgt_batch:
                seq.extend([PAD_IDX] * (tgt_batch_len - len(seq)))
            data.append((LongTensor(src_batch), LongTensor(tgt_batch)))
            src_batch = []
            tgt_batch = []
            src_batch_len = 0
            tgt_batch_len = 0
    fo.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, src_vocab, tgt_vocab

def test():
    print("cuda: %s" % CUDA)
    data, src_vocab, tgt_vocab = load_data()
    if VERBOSE:
        src_itow = [w for w, _ in sorted(src_vocab.items(), key = lambda x: x[1])]
        tgt_itow = [w for w, _ in sorted(tgt_vocab.items(), key = lambda x: x[1])]
    enc = encoder(len(src_vocab))
    dec = decoder(len(tgt_vocab))
    epoch = load_checkpoint(sys.argv[1], enc, dec) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    
    print('Generaint words...')
    for x, _ in tqdm(data):
        mask = mask_pad(x)
        pred = [[] for _ in range(BATCH_SIZE)]
            
        x = [src_itow[scalar(i)] for i in x[0]]
        x = torch.stack(list(map(word2vec, x)))
        enc_out = enc(x, mask)
        dec_in = LongTensor([SOS_IDX] * BATCH_SIZE).unsqueeze(1)
        for t in tqdm(range(1000)):
            dec_out = dec(enc_out, dec_in, mask).long()
            dec_in = torch.cat((dec_in, dec_out.data.topk(1)[1]), 1) # teacher forcing
            for i, j in enumerate(dec_out.data.topk(1)[1]):
                pred[i].append(scalar(j))
    
        with open('result.txt', 'a+') as RESULT:  
            for x, y in zip(x, pred):
                RESULT.write(''.join([tgt_itow[i] for i in y if i != PAD_IDX]) + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model vocab.src vocab.tgt testing_data" % sys.argv[0])
    test()
