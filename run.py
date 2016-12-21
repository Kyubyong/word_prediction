# -*- coding: utf-8 -*-
'''
Tokenizes English sentences using neural networks
Nov., 2016. Kyubyong.
'''

import sugartensor as tf
import numpy as np
from prepro import Hyperparams, load_data, load_charmaps
from train import ModelGraph
import codecs
import copy

def vectorize_input():
    '''Embeds and vectorize words in input corpus'''
    paras = [line for line in codecs.open('data/input.txt', 'r', 'utf-8').read().splitlines()]
    
    char2idx = load_charmaps()[0]
    
    xs = []
    for para in paras:
        x = []
        for char in para:
            if char in char2idx: 
                x.append(char2idx[char])
            else:
                x.append(char2idx["_"]) #"OOV"
        assert len(x) == len(para)
        xs.append(x)
    
    return paras, xs  
    # paras: [u'Summer is here ... ', 'Heat exhaustion is a relatively ...']
    # xs: [[34, 63, 55, 55, ...], [25, 47, 44, 64, 2, 47, ...]]

def main():  
    g = ModelGraph(is_train=False)
        
    with tf.Session() as sess:
        tf.sg_init(sess)

        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))
                     
        paras, xs = vectorize_input() 
        char2idx, idx2char = load_charmaps()
        space_idx = char2idx[' ']
        with codecs.open('data/output_model-006-333257.txt', 'w', 'utf-8') as fout:
            fout.write("{}\t{}\t{}\t{}\n".format("Characters", "Target Words", "Predictions", "Cumulative Keystroke Numbers", "Hits"))
            hits = 0
            stop_counting = False
            for para, x in zip(paras, xs):
                x = [0] * (Hyperparams.seqlen - 1) + x + [0]
                para = "_" * (Hyperparams.seqlen - 2) + " " + para
                
                chars, words = [], [] # words: the word that the char composes
                for word in para.split():
                    chars.append(" ")
                    words.append(word)
                    for char in word:
                        chars.append(char)
                        words.append(word)
                
                prefix = "" 
                for i, pair in enumerate(zip(chars, words)):
                    char, word = pair
                    if i > Hyperparams.seqlen-1:
                        ctx = x[i - Hyperparams.seqlen + 1:i] # indices of preceding 99 characters

                        tgt = x[i] # index of target character
                        
                        if char == " ": 
                            prefix = ""
                            stop_counting = False
                        else: 
                            prefix += char
                        
                        # predict characters
                        preds = ""
                        _ctx = copy.copy(ctx)
                        
                        j = 1
                        while j < 10:
                            logits = sess.run(g.logits, {g.x: np.expand_dims(_ctx, 0)}) #(1, 70)
                            pred = np.argmax(logits) #()
                              
                            if pred == space_idx:
                                break
                            else:
                                preds += idx2char[pred]
                            _ctx = _ctx[1:]
                            _ctx.append(pred.tolist())
                            j += 1
                            
                        if not stop_counting:
                            hits += 1   
                                 
                        predicted_word =  prefix + preds
                        if predicted_word == word: # If the prediction is correct, we stop counting until the next word.
                            stop_counting = True
                            
                        fout.write("{}\t{}\t{}\t{}\t{}\n".format(char, word, predicted_word, str(hits), str(stop_counting)))
                        

                            
                    
#                 current_prefix = ""   
#                 for i, char_targetword in enumerate(list_of_char_targetword_tuples):
#                     char, targetword = char_targetword
#                     
#                     ctx = x[i: i + Hyperparams.seqlen - 1] # indices of preceding 99 characters
#                     tgt = x[i + Hyperparams.seqlen - 1] # index of target character
#                     print char, targetword, tgt
#                 for i in range(len(para)):
#                     if i >= Hyperparams.seqlen-1:
#                         history = x[i - Hyperparams.seqlen + 1 :i]
#                         target = x[i]
#                         current_char = para[i-1]
#                         if current_char == " ": current_prefix = ""
#                         else: current_prefix += current_char
#                             
#                         # predict characters
#                         word_pred = ""
#                         x_ = np.copy(history)
#                          
#                         j = 1
#                         while j < 10:
#                             logits = sess.run(g.logits, {g.x: np.expand_dims(x_, 0)}) #(1, 70)
#                             pred = np.argmax(logits) #()
#                              
#                             if pred == space_idx:
#                                 break
#                             else:
#                                 word_pred += idx2char[pred]
#                             x_ = x_[1:]
#                             x_.append(pred.tolist())
#                             j += 1
#                          
#                         fout.write("{}\t{}\n".format(current_char, current_prefix+word_pred))
                                        
if __name__ == '__main__':
    main()
    print "Done"

