import sys
import re
import time
from model import *
from utils import *
from os.path import isfile
from torch.autograd import Variable

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

def train():
    print("cuda: %s" % CUDA)
    num_epochs = int(sys.argv[5])
    data, src_vocab, tgt_vocab = load_data()
    if VERBOSE:
        src_itow = [w for w, _ in sorted(src_vocab.items(), key = lambda x: x[1])]
        tgt_itow = [w for w, _ in sorted(tgt_vocab.items(), key = lambda x: x[1])]
    enc = encoder(len(src_vocab))
    dec = decoder(len(tgt_vocab))
    enc_optim = torch.optim.Adam(enc.parameters())
    dec_optim = torch.optim.Adam(dec.parameters())
    epoch = load_checkpoint(sys.argv[1], enc, dec) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print("training model...")
    with open('log.txt', 'a+') as LOG:
        for ei in range(epoch + 1, epoch + num_epochs + 1):
            ii = 0
            loss_sum = 0
            timer = time.time()
            for x, y in data:
                ii += 1
                loss = 0
                total_loss = 0
                enc.zero_grad()
                dec.zero_grad()
                mask = mask_pad(x)
                if VERBOSE:
                    pred = [[] for _ in range(BATCH_SIZE)]
                enc_out = enc(x, mask)
                dec_in = LongTensor([SOS_IDX] * BATCH_SIZE).unsqueeze(1)
                for t in range(y.size(1)):
                    dec_out = dec(enc_out, dec_in, mask)
                    loss = F.nll_loss(dec_out, y[:, t], size_average = False, ignore_index = PAD_IDX)
                    total_loss += loss.item()
                    loss.backward(retain_graph=True)
                    del loss

                    dec_in = torch.cat((dec_in, y[:, t].unsqueeze(1)), 1) # teacher forcing
                    if VERBOSE:
                        for i, j in enumerate(dec_out.data.topk(1)[1]):
                            pred[i].append(scalar(j))
                
                total_loss /= y.data.gt(0).sum().float() # divide by the number of unpadded tokens
                enc_optim.step()
                dec_optim.step()
                total_loss = scalar(total_loss)
                loss_sum += total_loss
                print("epoch = %d, iteration = %d, loss = %f" % (ei, ii, total_loss))
                del total_loss
            timer = time.time() - timer
            loss_sum /= len(data)
            LOG.write(f'epoch = {ei}, loss_sum={loss_sum}')
            if ei % SAVE_EVERY and ei != epoch + num_epochs:
                save_checkpoint("", None, None, ei, loss_sum, timer)
            else:
                if VERBOSE:
                    for x, y in zip(x, pred):
                        print(" ".join([src_itow[scalar(i)] for i in x if scalar(i) != PAD_IDX]))
                        print(" ".join([tgt_itow[i] for i in y if i != PAD_IDX]))
                save_checkpoint(filename, enc, dec, ei, loss_sum, timer)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model vocab.src vocab.tgt training_data num_epoch" % sys.argv[0])
    train()
