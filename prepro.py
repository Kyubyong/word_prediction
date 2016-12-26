#!/usr/bin/python2
# coding: utf-8
'''
Preprocessing.
'''

import numpy as np
import cPickle as pickle
import re

class Hyperparams:
    '''Hyper parameters'''
    batch_size = 16
    embed_dim = 200
    seqlen = 100 # We will predict next characters based on the previous 100 characters.
    
def prepro():
    '''Embeds and vectorize words in corpus'''
    
    # Make corpus. Note that the basic unit is paragraphs, not sentences.
    from nltk.corpus import reuters
    
    def clean_sent(sent):
        sent = sent.lower()
        sent = re.sub("[^ 0-9a-z\-.?!']", "~", sent) # "~" for unknown inputs
        sent = re.sub("([.?!])", r" \1", sent)
        return sent
    
    paras = []
    for para in reuters.paras(): # paragraph-wise
        
        sents = []
        for sent in para: # sentence-wise. FYI, 54716 sentences in total
            _sent = " ".join(sent)
            _sent = clean_sent(_sent)
            sents.append(_sent)
        _para = " ".join(sents)
        if len(_para) < 1000:
            paras.append(_para)
    
    char2idx, idx2char = load_charmaps()
     
    print "# Vectorize"
    xs = [] # vectorized sentences
    for para in paras:
        x, y = [], []
        for char in para:
            if char in char2idx:
                x.append(char2idx[char])
            else:
                x.append(char2idx["~"]) #"OOV"
        xs.append( [0] * (1000 - len(x)) + x ) # zero pre-padding
  
    print "# Convert to 2d-arrays"
    X = np.array(xs)
    
    print "X.shape =", X.shape
    np.save('data/X', X)
             
def load_charmaps():
    vocab = "@ abcdefghijklmnopqrstuvwxyz0123456789.-'~" # @:empty, ~: unknown
    char2idx = {char:idx for idx, char in enumerate(vocab)}
    idx2char = {idx:char for idx, char in enumerate(vocab)}  
    
    return char2idx, idx2char  

def load_data():
    '''Loads vectorized input training data
    '''
    return np.load('data/X.npy') # (11887,)

if __name__ == '__main__':
    prepro()
    print "Done"        
