import jieba
from gensim.models import FastText
from tqdm import tqdm

jieba.load_userdict('dict-traditional.txt')
jieba.load_userdict('sky_dragon_name.txt')

with open('sky_dragon_and_lcstcs_head10000.csv') as IN:
    words = []
    for line in tqdm(IN.readlines()):
        rows = line.split(',')
        words.append(jieba.lcut(rows[0]))
        words.append(jieba.lcut(rows[1]))

    model = FastText(words, size=512, window=10, min_count=1, sg=1, iter=100)
    model.save('wordvec/skipgram.model')