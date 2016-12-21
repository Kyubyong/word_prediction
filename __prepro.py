#/usr/bin/python2
# coding: utf-8
'''
Preprocessing.
'''

import numpy as np
import cPickle as pickle
import re
import codecs

class Hyperparams:
    '''Hyper parameters'''
    dataset_fpath = '../../datasets/english_news/english_news.txt'
    batch_size = 16
    embed_dim = 200
    seqlen = 100 # We will predict next characters based on the previous 99 characters.

def load_charmaps():
    vocab = "@ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-'~" # @:empty, ~: unknown
    char2idx = {char:idx for idx, char in enumerate(vocab)}
    idx2char = {idx:char for idx, char in enumerate(vocab)}  
    
    return char2idx, idx2char      
    
def prepro():
    '''Embeds and vectorize words in corpus'''
    
    from nltk.tokenize import word_tokenize
    
    # Make corpus. Note that the basic unit is paragraphs, not sentences.
    def clean_text(text):
        text = re.sub("[^ 0-9A-Za-z\-']", "", text)
        return text
    
    paras = []
    maxlen = 700
    with codecs.open(Hyperparams.dataset_fpath, 'r', 'utf-8') as fin:
        num_line = 0
        while 1:
            num_line += 1
            line = fin.readline()
            if not line: break
            if num_line > 200000: break
            
            line = clean_text(line)
            if 10 < len(line) < maxlen:
                paras.append(line)
    
    print "# Vectorize"
    char2idx, idx2char = load_charmaps()
    xs = [] # vectorized sentences
    for para in paras:
        x, y = [], []
        for char in para:
            if char in char2idx:
                x.append(char2idx[char])
            else:
                x.append(char2idx["~"]) #"OOV"
        xs.append( [0] * (maxlen - len(x)) + x ) # zero pre-padding
  
    print "# Convert to 2d-arrays"
    X = np.array(xs)
    
    print "X.shape =", X.shape
  
    np.save('data/X', X)
             
def load_data():
    '''Loads vectorized input training data
    '''
    return np.load('data/X.npy') # (11887,)

if __name__ == '__main__':
    prepro()
    load_data()
    print "Done"        
