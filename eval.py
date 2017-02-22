# -*- coding: utf-8 -*-
'''
Evaluation.
We simulate typing in the mobile environment.
In most smartphones, words are suggested on the top of the keyboard area
so the user can choose their intended word before finishing his typing.
Suppose the user always select the word if it is on the center of suggestion bar
(the top candidate). We call it a `responsive keystroke (rk)`. We can evaluate the performance
of the predictive engine by counting the number of responsive keystrokes
compared to the number of original keystrokes, which we call `full keystrokes (fk)`.
Finally, the evaluation is conducted by calculating how many numbers of keystrokes
were saved by the predictive engine. We call the metric `keystroke savings rate (ksr)`.

ksr = (fk - rk) / fk
 
For the test 64 English paragraphs,
the generated model predicts a word per letter including a space.
'''
from __future__ import print_function
import sugartensor as tf
import numpy as np
from prepro import *
from train import ModelGraph
import codecs

def main(): 
    g = ModelGraph(mode="test")
        
    with tf.Session() as sess:
        tf.sg_init(sess)

        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train'))
        print("Restored!")
        mname = open('asset/train/checkpoint', 'r').read().split('"')[1] # model name
        
        X, Y = load_test_data()
        char2idx, idx2char = load_char_vocab()
        word2idx, idx2word = load_word_vocab()
        
        results = []
        rk = 0
        num_para = 1
        num_char = 1
        for x, y in zip(X, Y):
            stop_counting = False
            x = np.concatenate( (np.zeros((Hyperparams.seqlen-1,)), 
                                 x[-np.count_nonzero(x):]))# lstrip and zero-pad
            
            para = "".join([idx2char[idx] for idx in x])
            
            chars, targets = [], [] # targets: the word that the char composes
            for word in "".join(para).split():
                chars.append(" ")
                targets.append(word)
                for char in word:
                    chars.append(char)
                    targets.append(word)
            
            prefix = "" 
            preds = set()
            for i, char_target in enumerate(zip(chars, targets)):
                char, target = char_target
                oov = ""
                if target not in word2idx: 
                    oov = "oov"
                
                if i > Hyperparams.seqlen:
                    ctx = np.array(x[i - Hyperparams.seqlen:i], np.int32) # 
                    
                    if char == " ":
                        stop_counting = False
                        preds = set()
                        
                    if not stop_counting:
                        logits = sess.run(g.logits, {g.x: np.expand_dims(ctx, 0)}) #(1, 20970)
                        while 1:
                            pred = np.argmax(logits, -1)[0] # (1,)
                            if pred in preds:
                                logits[:, pred] = -100000000
                            else:
                                break
                        
                        rk += 1
                        
                        predword = idx2word.get(pred)    
                        if predword == target: # S
                            stop_counting = True
                        preds.add(pred)
                    
                    results.append(u"{},{},{},{},{},{},{}".format(num_char, num_para, char, target, oov, predword, rk) )
                    num_char += 1
            
            num_para += 1
            
        with codecs.open('data/output_{}_rk_{}.csv'.format(mname, rk), 'w', 'utf-8') as fout:
            fout.write("\n".join(results))
                                        
if __name__ == '__main__':
    main()
    print("Done")

