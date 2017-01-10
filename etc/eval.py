# -*- coding: utf-8 -*-
'''
Word Prediction
Jan. 2017. Kyubyong.
'''

import sugartensor as tf
import numpy as np
from prepro import Hyperparams, load_test_data, load_vocab
from train import ModelGraph
import codecs
import copy


def main():  
    g = ModelGraph(is_train=False)
        
    with tf.Session() as sess:
        tf.sg_init(sess)

        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))
        mname = open('asset/train/ckpt/checkpoint', 'r').read().split('"')[1] # model name
                     
        paras, xs = load_test_data() 
        char2idx, idx2char = load_vocab()
        with codecs.open('data/output_{}.txt'.format(mname), 'w', 'utf-8') as fout:
            fout.write("{}\t{}\t{}\t{}\n".format("Characters", "Target Words", "Predictions", "# Cumulative Keystrokes", "Hits"))
            hits = 0
            stop_counting = False
            for para, x in zip(paras, xs):
                x = [0] * (Hyperparams.ctxlen-1) + [1] + x + ([len(char2idx)] * Hyperparams.predlen)
                para = "_" * (Hyperparams.ctxlen-1) + " " + para + ("_" * Hyperparams.predlen)
                
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
                    
                    if Hyperparams.ctxlen < i < (len(chars) - Hyperparams.predlen):
                        if char == " ":
                            prefix = ""
                        prefix += char
                        ctx = np.array(x[i - Hyperparams.ctxlen:i], np.int64) # indices of preceding 100 characters
                        _x = np.concatenate((ctx, np.ones((Hyperparams.predlen,))*len(char2idx)))
                        _x = np.expand_dims(_x, 0)
                        pred_word = ''
                        for j in range(10):
                            logits = sess.run(g.logits, {g.x: _x}) #(1, 60, 70)
#                             preds = np.squeeze(np.argmax(logits, -1))[-Hyperparams.predlen:] # (60,)
                            pred = np.squeeze(np.argmax(logits, -1))[50+j] # (60,)
                            if pred == 1:
                                print char, word, prefix + pred_word
                                break
                            else:
                                pred_word += idx2char[pred]
                                _x[:, 50+j] = pred
#                             pred_word = ""
#                             for pred in preds:
#                                 pred_word += idx2char.get(pred, "*")
#                                 if pred == 1: # space
#                                     break
#                             print char, word, pred_word
#                         
#                             
# #                         def lstrip(ctx):
# #                             '''Replace elements before the first space with zeros.'''
# #                             first_ind_of_1 = np.where(ctx == 1)[0][0] # 1 means space.
# #                             ctx = np.concatenate((np.zeros((first_ind_of_1)), ctx[first_ind_of_1:]))
# #                             return ctx
# #                         
# #                         ctx = lstrip(ctx)
# #                         
#                         tgt = x[i] # index of target character
#                         
#                         if char == " ": 
#                             prefix = ""
#                             stop_counting = False
#                         else: 
#                             prefix += char
#                         
#                         # predict characters
#                         preds = ""
#                         _ctx = copy.copy(ctx)
#                         
#                         j = 1
#                         while j < 10:
#                             logits = sess.run(g.logits, {g.x: np.expand_dims(_ctx, 0)}) #(1, 100)
#                             pred = np.argmax(logits) #()
#                               
#                             if pred == 1: # space
#                                 break
#                             else:
#                                 preds += idx2char[pred]
#                             _ctx = _ctx[1:]
#                             _ctx = np.append(_ctx, pred)
#                             _ctx = lstrip(_ctx)
#                             j += 1
#                             
#                         if not stop_counting:
#                             hits += 1   
#                                  
#                         predicted_word =  prefix + preds
#                         if predicted_word == word: # If the prediction is correct, we stop counting until the next word.
#                             stop_counting = True
#                             
#                         fout.write("{}\t{}\t{}\t{}\t{}\n".format(char, word, predicted_word, str(hits), str(stop_counting)))
                        

                            
                    
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

