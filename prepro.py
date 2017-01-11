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
import lxml.etree as ET
import regex
from nltk.tokenize import word_tokenize

class Hyperparams:
    '''Hyper parameters'''
    batch_size = 64
    embed_dim = 300
    seqlen = 50  # We will predict the next character based on the previous 50 characters.

def load_vocab():
    vocab = "EUS abcdefghijklmnopqrstuvwxyz0123456789-.,?!'" # E: Empty, U:Unknown
    char2idx = {char:idx for idx, char in enumerate(vocab)}
    idx2char = {idx:char for idx, char in enumerate(vocab)}  
    
    return char2idx, idx2char      

def create_data():
    print("# Vectorize")
    char2idx, idx2char = load_vocab()
    lines = codecs.open('data/en_wikinews.txt', 'r', 'utf-8').read().splitlines()
    xs = [] # vectorized sentences
    for line in lines:
        x = []
        for char in line:
            if char in char2idx:
                x.append(char2idx[char])
            else: # Unknown
                x.append(1)
        if len(x) <= 1000: xs.append([0] * (1000 - len(x)) + x)
  
    print("# Convert to 2d-arrays")
    X = np.array(xs)
    
    print(X.shape)
    np.save('data/X.npy', X) # (197011, 1000) 

def load_train_data():
    return np.load('data/X.npy')[:-Hyperparams.batch_size]
def load_test_data():
#     return np.load('data/X.npy')[-Hyperparams.batch_size:]
    return np.load('data/X.npy')[-1]
                 


if __name__ == '__main__':
    create_data()
    print("Done")        