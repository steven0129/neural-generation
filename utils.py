import re
import jieba
import functools
from model import *

jieba.load_userdict('dict-traditional.txt')
jieba.load_userdict('sky_dragon_name.txt')

def normalize(x, y):
    if y== '，' or y == '。' or y == '：' or y == '「' or y == '」' or y == '！' or y == '？' or y == '‘' or y == '’':
        return x + y
    
    return y

def tokenize(x, unit):
    if unit == "char":
        x = re.sub(" ", "", x)
        return list(x)
    if unit == "word":
        result = []
        for xx in jieba.cut(x):
            if xx == '，' or xx == '。' or xx == '：' or xx == '「' or xx == '」' or xx == '！' or xx == '？' or xx == '‘' or xx == '’':
                if len(result) != 0:
                    result[-1] += xx
            else:
                result.append(xx)
        
        return result

def load_vocab(filename, ext):
    print("loading vocab.%s..." % ext)
    vocab = {}
    fo = open(filename)
    for line in fo:
        line = line.strip()
        vocab[line] = len(vocab)
    fo.close()
    return vocab

def load_checkpoint(filename, enc = None, dec = None):
    print("loading model...")
    checkpoint = torch.load(filename)
    enc.load_state_dict(checkpoint["encoder_state_dict"])
    dec.load_state_dict(checkpoint["decoder_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, enc, dec, epoch, loss, time):
    print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
    if filename and enc and dec:
        print("saving model...")
        checkpoint = {}
        checkpoint["encoder_state_dict"] = enc.state_dict()
        checkpoint["decoder_state_dict"] = dec.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        print("saved model at epoch %d" % epoch)
