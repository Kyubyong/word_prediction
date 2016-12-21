#/usr/bin/python2
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
    seqlen = 100 # We will predict next characters based on the previous 99 characters.
    
def prepro():
    '''Embeds and vectorize words in corpus'''
    
    # Make corpus. Note that the basic unit is paragraphs, not sentences.
    from nltk.corpus import reuters
    
    def clean_sent(sent):
        sent = re.sub("[^ 0-9A-Za-z\-\.\?!\']", "_", sent) # "_" for unknown inputs
        return sent
    
    paras = []
    maxlen = 0
    for para in reuters.paras(): # paragraph-wise
        new = []
        for sent in para: # sentence-wise. FYI, 54716 sentences in total
            sent_ = " ".join(sent)
            sent_ = clean_sent(sent_)
            new.append(sent_)
        new = " ".join(new)
        paras.append(new)
        maxlen = max(maxlen, len(new))
    
    print "# Create Vocabulary"
    vocab = ["<EMP>"] + list(set("".join(paras)))    
     
    print "# Create character maps"   
    char2idx = {char:idx for idx, char in enumerate(vocab)}
    idx2char = {idx:char for idx, char in enumerate(vocab)}
    
    print "vocabulary size =", len(char2idx) 
    pickle.dump((char2idx, idx2char), open('data/charmaps.pkl', 'wb'))
     
    print "# Vectorize"
    xs = [] # vectorized sentences
    for para in paras:
        x, y = [], []
        for char in para:
            if char in char2idx:
                x.append(char2idx[char])
            else:
                x.append(char2idx["_"]) #"OOV"
        xs.append( [0] * (maxlen - len(x)) + x ) # zero pre-padding
  
    print "# Convert to 2d-arrays"
    X = np.array(xs)
    
    print "X.shape =", X.shape
  
    np.save('data/X', X)
             
def load_charmaps():
    '''Loads character dictionaries'''
    char2idx, idx2char = pickle.load(open('data/charmaps.pkl', 'rb'))
    return char2idx, idx2char

def load_data():
    '''Loads vectorized input training data
    '''
    return np.load('data/X.npy') # (11887,)

if __name__ == '__main__':
    prepro()
    load_data()
    print "Done"        
