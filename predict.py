import sys
import re
from model import *
from utils import *
from gensim.models import FastText

fastText = FastText.load_fasttext_format('wordvec/skipgram.bin')

def word2vec(x):
    try:
        return torch.from_numpy(fastText.wv[x])
    except:
        return torch.rand(512)

def load_model():
    src_vocab = load_vocab(sys.argv[2], "src")
    tgt_vocab = load_vocab(sys.argv[3], "tgt")
    tgt_vocab = [word for word, _ in sorted(tgt_vocab.items(), key = lambda x: x[1])]
    enc = encoder(len(src_vocab))
    dec = decoder(len(tgt_vocab))
    enc.eval()
    dec.eval()
    print(enc)
    print(dec)
    load_checkpoint(sys.argv[1], enc, dec)
    return enc, dec, src_vocab, tgt_vocab

def run_model(enc, dec, tgt_vocab, data):
    batch = []
    pred = [[] for _ in range(BATCH_SIZE)]
    z = len(data)
    eos = [0 for _ in range(z)] # number of EOS tokens in the batch
    while len(data) < BATCH_SIZE:
        data.append(["", [EOS_IDX], []])
    data.sort(key = lambda x: len(x[1]), reverse = True)
    batch_len = len(data[0][1])
    batch = LongTensor([x[1] for x in data])
    # batch = LongTensor([x[1] + [PAD_IDX] * (batch_len - len(x[1])) for x in data])
    mask = mask_pad(batch)
    enc_out = enc(batch, mask)
    # print(enc_out)
    dec_in = LongTensor([SOS_IDX] * BATCH_SIZE).unsqueeze(1)
    t = 0
    
    for t in range(10):
        dec_out = dec(enc_out, dec_in, mask)
        dec_out = dec_out.long()
        dec_in = torch.cat((dec_in, dec_out.data.topk(5)[1][4]), 1)
        # print(dec_in)
        # print(dec_out.data.topk(1)[1])
        for i, j in enumerate(dec_out.data.topk(1)[1]):
            pred[i].append(scalar(j))
    
    for y in pred:
        for i in y:
            if i != PAD_IDX:
                print(' '.join([tgt_vocab[i]]))
        # print(" ".join([tgt_vocab[i] for i in y if i != PAD_IDX]))
    return data[:z]

def predict():
    data = []
    pred = [[] for _ in range(BATCH_SIZE)]
    enc, dec, src_vocab, tgt_vocab = load_model()
    with open(sys.argv[4]) as fo:
        for line in fo.readlines():
            line = line.strip()
            x = list(filter(lambda x: x != ' ', tokenize(line, "word")))
            mask = mask_pad(LongTensor([src_vocab[i] for i in x] + [EOS_IDX]))
            x = torch.stack(list(map(word2vec, x + ['<EOS>'])))
            enc_out = enc(x, mask)
            dec_in = LongTensor([SOS_IDX] * BATCH_SIZE).unsqueeze(1)
            
            for t in range(1000):
                dec_out = dec(enc_out, dec_in, mask)
                dec_out = dec_out.long()
                dec_in = torch.cat((dec_in, dec_out.data.topk(1)[1]), 1)
                for i, j in enumerate(dec_out.data.topk(1)[1]):
                    pred[i].append(scalar(j))

            result = []
            for y in pred:
                for i in y:
                    if i != PAD_IDX:
                        result.append(tgt_vocab[i])

            print(''.join(result))
            # if len(data) == BATCH_SIZE:
            #     result = run_model(enc, dec, tgt_vocab, data)
            #     for x in result:
            #         pass
            #     data = []

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model vocab.src vocab.tgt test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    predict()
