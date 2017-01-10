#/usr/bin/python2
# coding: utf-8
'''
Preprocessing.
'''

import numpy as np
import cPickle as pickle
import re
import codecs
from nltk.tokenize import word_tokenize, sent_tokenize

class Hyperparams:
    '''Hyper parameters'''
    dataset_fpath = '../../datasets/english_news/english_news.txt'
    batch_size = 64
    embed_dim = 200
    ctxlen = 50 
    predlen = 10 # We will predict the next 10 characters based on the previous 50 characters.

def load_vocab():
    vocab = "@ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-.,?!'" # @:empty
    char2idx = {char:idx for idx, char in enumerate(vocab)}
    idx2char = {idx:char for idx, char in enumerate(vocab)}  
    
    return char2idx, idx2char      

def clean(text):
    '''
    Lowercases the first word except `I`.
    '''
    try:
        ret = []
        text = re.sub("[^ 0-9A-Za-z\-.,?!']", "", text)
        sents = sent_tokenize(text)
        for sent in sents:
            words = word_tokenize(sent)
            first_word = words[0]
            if first_word != "I":
                first_word = first_word.lower()
            if len(words) > 1:
                words = [first_word] + words[1:]
            else: 
                return ""
            ret.extend(words)
        return " ".join(ret)
    except:
        pass
    
    return ""
    
def create_train_data():
    '''Embeds and vectorize words in corpus'''
    
    paras = []
    with codecs.open(Hyperparams.dataset_fpath, 'r', 'utf-8') as fin:
        num_line = 0
        while 1:
            num_line += 1
            line = fin.readline()
            if not line: break
            
            line = clean(line)
            if 900 < len(line) < 1000:
                paras.append(line)
            
            if num_line % 1000 == 0: print num_line
    
    print "# Vectorize"
    char2idx, idx2char = load_vocab()
    xs = [] # vectorized sentences
    for para in paras:
        x = []
        for char in para:
            x.append(char2idx[char])
        xs.append( [0] * (1000 - len(x)) + x ) # zero pre-padding
  
    print "# Convert to 2d-arrays"
    X = np.array(xs)
    
    print "X.shape =", X.shape
  
    np.save('data/X', X)
             
def load_train_data(mode="train"):
    '''Loads vectorized input training data
    '''
    if mode == "train":
        return np.load('data/X.npy')[:-Hyperparams.batch_size] # (11887,)
    else: #val
        return np.load('data/X.npy')[-Hyperparams.batch_size:] # (11887,)

def load_test_data():
    '''Embeds and vectorize words in input corpus'''
    paras = [line for line in codecs.open('data/input.txt', 'r', 'utf-8').read().splitlines()]
    
    char2idx = load_vocab()[0]
    
    xs = []
    for para in paras:
        x = []
        for char in para:
            if char in char2idx: 
                x.append(char2idx[char])
            else:
                x.append(char2idx["~"]) #"OOV"
        assert len(x) == len(para)
        xs.append(x)
    
    return paras, xs  
    # paras: [u'Summer is here ... ', 'Heat exhaustion is a relatively ...']
    # xs: [[34, 63, 55, 55, ...], [25, 47, 44, 64, 2, 47, ...]]

if __name__ == '__main__':
    create_train_data()
    print "Done"        