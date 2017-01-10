# -*- coding: utf-8 -*-
'''
Word Prediction
Jan. 2017. Kyubyong.
'''
from __future__ import print_function
from prepro import Hyperparams, load_data, load_vocab
import sugartensor as tf

def get_batch_data(mode="train"):
    '''Makes batch queues from the data.
    
    Args:
      mode: A string. Either 'train', 'val', or 'test' 
    Returns:
      A Tuple of X_batch (Tensor), Y_batch (Tensor), and number of batches (int).
      X_batch and Y_batch have of the shape [batch_size, maxlen].
    '''
    # Load data
    X = load_data(mode=mode)

    # Create Queues
    x_q, = tf.train.slice_input_producer([tf.convert_to_tensor(X, tf.int64)]) # (1000,)
    
    # Lstrip zeros
    zeros = tf.equal(x_q, tf.zeros_like(x_q)).sg_int().sg_sum()
    x_q = x_q[zeros:] 
    
    # Initial Padding
    x_q = tf.concat(0, [tf.zeros([Hyperparams.seqlen], tf.int64), x_q]) # 50 zero-padding
    
    # Random crop
    x_q = tf.random_crop(x_q, [Hyperparams.seqlen + 1]) # (50 + 1,)
#     
    # create batch queues
    x = tf.train.shuffle_batch([x_q],
                              num_threads=32,
                              batch_size=Hyperparams.batch_size, 
                              capacity=Hyperparams.batch_size*64,
                              min_after_dequeue=Hyperparams.batch_size*32, 
                              allow_smaller_final_batch=False)  # (16, 100)
    
    num_batch = len(X) // Hyperparams.batch_size
    
    return x, num_batch

class ModelGraph():
    '''Builds a model graph'''
    def __init__(self, mode="train"):
        '''
        Args:
          mode: A string. Either "train" , "val", or "test"
        '''
        self.char2idx, self.idx2char = load_vocab()
        
        if mode == "train":
            self.x, self.num_batch = get_batch_data() # (64, 51) int64
            self.x, self.y = self.x[:, :Hyperparams.seqlen], self.x[:, Hyperparams.seqlen] # (64, 50) int64, (64, 1) int64
        else:
            self.x = tf.placeholder(tf.int64, [None, Hyperparams.seqlen])
        
        # make embedding matrix for input characters
        self.emb_x = tf.sg_emb(name='emb_x', voca_size=len(self.char2idx), dim=Hyperparams.embed_dim)
        self.enc = self.x.sg_lookup(emb=self.emb_x)
        
        with tf.sg_context(size=5, act='relu', bn=True):
            for _ in range(20):
                dim = self.enc.get_shape().as_list()[-1]
                self.enc += self.enc.sg_conv1d(dim=dim) # (64, 50, 300) float32
                
        # final fully convolution layer for softmax
        self.logits = self.enc.sg_conv1d(size=1, dim=len(self.char2idx)) # (64, 50, 44) float32
        self.logits = tf.reduce_mean(self.logits, reduction_indices=[1], keep_dims=False) # (64, 44) float32
        
        if mode == "train":
            self.ce = self.logits.sg_ce(target=self.y, mask=False, one_hot=False)

def train():
    g = ModelGraph()
    print("Graph loaded!")

    tf.sg_train(lr_reset=True, log_interval=10, loss=g.ce, eval_metric=[], max_ep=2000, 
                save_dir='asset/train', early_stop=False, max_keep=10, ep_size=g.num_batch)
     
if __name__ == '__main__':
    train(); print("Done")
