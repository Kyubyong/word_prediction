# -*- coding: utf-8 -*-
'''
Training.
'''
from __future__ import print_function
from prepro import *
import sugartensor as tf
import random

def q_process(t1, t2):
    '''
    Processes each training sample so that it fits in the queue.
    '''
    # Lstrip zeros
    zeros = tf.equal(t1, tf.zeros_like(t1)).sg_int().sg_sum()
    t1 = t1[zeros:] 
    t2 = t2[zeros:]

    # zero-PrePadding
    t1 = tf.concat([tf.zeros([Hyperparams.seqlen-1], tf.int32), t1], 0)# 49 zero-prepadding
    t2 = tf.concat([tf.zeros([Hyperparams.seqlen-1], tf.int32), t2], 0)# 49 zero-prepadding
    # radom crop    
    stacked = tf.stack((t1, t2))
    cropped = tf.random_crop(stacked, [2, Hyperparams.seqlen])
    t1, t2 = cropped[0], cropped[1]
    
    t2 = t2[-1]

    return t1, t2

def get_batch_data():
    '''Makes batch queues from the data.
    '''
    # Load data
    X, Y = load_train_data() # (196947, 1000) int64

    # Create Queues
    x_q, y_q = tf.train.slice_input_producer([tf.convert_to_tensor(X, tf.int32),
                                          tf.convert_to_tensor(Y, tf.int32)]) # (1000,) int32
    
    x_q, y_q = q_process(x_q, y_q) # (50,) int32, () int32

    # create batch queues
    x, y = tf.train.shuffle_batch([x_q, y_q],
                              num_threads=32,
                              batch_size=Hyperparams.batch_size, 
                              capacity=Hyperparams.batch_size*64,
                              min_after_dequeue=Hyperparams.batch_size*32, 
                              allow_smaller_final_batch=False)
    
    num_batch = len(X) // Hyperparams.batch_size

    return x, y, num_batch # (64, 50) int32, (64, 50) int32, ()

class ModelGraph():
    '''Builds a model graph'''
    def __init__(self, mode="train"):
        '''
        Args:
          mode: A string. Either "train" or "test"
        '''
        self.char2idx, self.idx2char = load_char_vocab()
        self.word2idx, self.idx2word = load_word_vocab()
        
        if mode == "train":
            self.x, self.y, self.num_batch = get_batch_data() 
        else:
            self.x = tf.placeholder(tf.int32, [None, Hyperparams.seqlen])
        
        self.emb_x = tf.sg_emb(name='emb_x', voca_size=len(self.char2idx), dim=Hyperparams.embed_dim)
        self.enc = self.x.sg_lookup(emb=self.emb_x)
        
        with tf.sg_context(size=5, act='relu', bn=True):
            for _ in range(20):
                dim = self.enc.get_shape().as_list()[-1]
                self.enc += self.enc.sg_conv1d(dim=dim) # (64, 50, 300) float32
        
        self.enc = self.enc.sg_conv1d(size=1, dim=len(self.word2idx), act='linear', bn=False) # (64, 50, 21293) float32
#         self.logits = self.enc.sg_mean(dims=[1], keep_dims=False) # (64, 21293) float32
        
        # Weighted Sum. Updated on Feb. 15, 2017.
        def make_weights(size):
            weights = tf.range(1, size+1, dtype=tf.float32)
            weights *= 1. / ((1 + size) * size // 2)
            weights = tf.expand_dims(weights, 0)
            weights = tf.expand_dims(weights, -1)
            return weights
        
        self.weights = make_weights(Hyperparams.seqlen) # (1, 50, 1)
        self.enc *= self.weights # Broadcasting
        self.logits = self.enc.sg_sum(axis=[1], keep_dims=False) # (64, 21293)

        if mode == "train":
            self.ce = self.logits.sg_ce(target=self.y, mask=False, one_hot=False)
            self.istarget = tf.not_equal(self.y, tf.ones_like(self.y)).sg_float() # 1: Unkown   
            self.reduced_loss = ((self.ce * self.istarget).sg_sum()) / (self.istarget.sg_sum() + 1e-5)
            tf.sg_summary_loss(self.reduced_loss, "reduced_loss")
            
def train():
    g = ModelGraph()
    print("Graph loaded!")

    tf.sg_train(optim="Adam", lr=0.00001, lr_reset=True, loss=g.reduced_loss, eval_metric=[], max_ep=20000, 
                save_dir='asset/train', early_stop=False, ep_size=g.num_batch)
     
if __name__ == '__main__':
    train(); print("Done")
