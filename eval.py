# -*- coding: utf-8 -*-
'''
Word Prediction
Jan. 2017. Kyubyong.
'''
from __future__ import print_function
import sugartensor as tf
import numpy as np
from prepro import Hyperparams, load_data, load_vocab, load_test_data
from train import ModelGraph
import codecs
import copy

def main():  
    g = ModelGraph(mode="test")
        
    with tf.Session() as sess:
        tf.sg_init(sess)

        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))
        mname = open('asset/train/ckpt/checkpoint', 'r').read().split('"')[1] # model name
        
#         X = load_data(mode="test")
        X = load_test_data()
        char2idx, idx2char = load_vocab()
        
        with codecs.open('data/output_{}.csv'.format(mname), 'w', 'utf-8') as fout:
            fout.write("{},{},{},{}\n".format("Characters", "Target Words", "Predictions", "# Responsive Keystrokes"))
            rk = 0
            stop_counting = False
            for x in X:
                x = np.concatenate( (np.zeros((Hyperparams.seqlen,)), x[-np.count_nonzero(x):]) ) # lstrip and zero-pad
                para = "".join([idx2char[idx] for idx in x])
                
                chars, words = [], [] # words: the word that the char composes
                for word in "".join(para).split():
                    chars.append(" ")
                    words.append(word.strip("_"))
                    for char in word:
                        chars.append(char)
                        words.append(word.strip("_"))
                
                prefix = "" 
                for i, char_word in enumerate(zip(chars, words)):
                    char, word = char_word
                    
                    if i > Hyperparams.seqlen:
                        ctx = np.array(x[i - Hyperparams.seqlen:i], np.int64) # 
                        
                        if char == " ":
                            prefix = ""
                            stop_counting = False
                        else: 
                            prefix += char  
                        
                        # predict characters      
                        suffix = '' # the latter part of word completion 
                        _ctx = copy.copy(ctx)
                        
                        j = 1
                        while j < 15: # We predict next 15 characters at most.
                            logits = sess.run(g.logits, {g.x: np.expand_dims(_ctx, 0)}) # (1, 44) float32
                            pred = np.argmax(np.squeeze(logits)).tolist()
                            
                            if pred == 1: # space
                                break
                            else:
                                suffix += idx2char[pred]
                            
                            _ctx = _ctx[1:]
                            _ctx = np.append(_ctx, pred)
                            j += 1
                        
                        if not stop_counting:
                            rk += 1
                        
                        pred_word = prefix + suffix
                        if pred_word == word:
                            stop_counting = True
                        
                        fout.write("{},{},{},{}\n".format(char, word, pred_word, str(rk)))    
                                        
if __name__ == '__main__':
    main()
    print("Done")

