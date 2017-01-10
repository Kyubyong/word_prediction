#/usr/bin/python2
# coding: utf-8
'''
Preprocessing.
'''
from __future__ import print_function
import numpy as np
import pickle
import re
import codecs

class Hyperparams:
    '''Hyper parameters'''
    batch_size = 64
    embed_dim = 300
    seqlen = 50  # We will predict the next character based on the previous 50 characters.

def load_vocab():
    vocab = "_ abcdefghijklmnopqrstuvwxyz0123456789-.,?!'" # _:empty
    char2idx = {char:idx for idx, char in enumerate(vocab)}
    idx2char = {idx:char for idx, char in enumerate(vocab)}  
    
    return char2idx, idx2char      

def create_data():
    '''Embeds and vectorize words in corpus'''
    from nltk.corpus import reuters
    
    def clean_sent(sent):
        sent = sent.lower()
        sent = re.sub("[^ 0-9a-z\-.?!']", " ", sent)
        sent = re.sub(" {2,}", " ", sent) # squeeze spaces
        return sent

    paras = []
    maxlen = 0
    for sents in reuters.paras(): # paragraph-wise
        para = ""
        for words in sents: # sentence-wise. FYI, 54716 sentences in total
            sent = " ".join(words)
            sent = clean_sent(sent)
            para += sent + " "
        para = para.strip()
        paras.append(para)
        maxlen = max(maxlen, len(para))
    
    print("# Vectorize")
    char2idx, idx2char = load_vocab()
    xs = [] # vectorized sentences
    for para in paras:
        x = []
        for char in para:
            x.append(char2idx[char])
        xs.append( [0] * (maxlen - len(x)) + x ) # zero pre-padding
  
    print("# Convert to 2d-arrays")
    X = np.array(xs)
    
    print("X.shape =", X.shape)
    np.save('data/X.npy', X) # (11887, 11648)

def load_test_data():
    char2idx, idx2char = load_vocab()
    s = """summer is here and though it is the time for outdoor fun and play , it is also the time to start thinking about protecting ourselves from extreme heat . extreme heat can cause serious health problems . but , you can do something about it . heat exhaustion is a relatively common response to intense heat and can cause symptoms such as headaches , dizziness and fainting . but it can also lead to heatstroke , which must be treated medically . heat stress is especially harmful for the elderly , small children and people who already suffer from illnesses such as heart disease , diabetes or hypertension . however , anyone should be careful and not do heavy physical activity when the temperature goes above 35 degrees . we can do many things to protect ourselves from the dangers of extreme heat . we should stay out of the sun , drink plenty of liquids and wear light clothing . also , avoid drinks that contain alcohol or sugar . actually , they will cause you to lose body fluids . i hope you can all enjoy your summer days . just play it safe ."""
    s = [char2idx[char] for char in s]
    S = np.array(s)
    return np.expand_dims(S, 0)
        
                 
def load_data(mode="train"):
    '''Loads vectorized input training data
    '''
    if mode == "train":
        return np.load('data/X.npy')[:-1]
    else: # test
        return np.load('data/X.npy')[-1]

if __name__ == '__main__':
    create_data()
    print("Done")        