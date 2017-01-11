# -*- coding: utf-8 -*-
'''
Word Prediction
Jan. 2017. Kyubyong.
'''
from __future__ import print_function
from prepro import Hyperparams, load_train_data, load_vocab
import sugartensor as tf

def label_process(tensor):
    '''
    a = np.array([1, 1, 2, 3, 4, 2, 2, 2,2,2,2,2], np.int32)
    aa = tf.convert_to_tensor(a)
    
    => [1 1 2 2 0 0 0 0 0 0 0 0]
    '''
    first_space_ind = tf.to_int32(tf.where(tf.equal(tensor, 3))[0][0]) # 3: space
    zero_postpadding = tf.zeros(tensor.get_shape()-first_space_ind-1, tf.int32)
    head = tensor[:first_space_ind]
    out = tf.concat(0, (head, [2], zero_postpadding)) # 2: S
    
    return out

def get_batch_data():
    '''Makes batch queues from the data.
    
    Args:
      mode: A string. Either 'train', 'val', or 'test' 
    Returns:
      A Tuple of X_batch (Tensor), Y_batch (Tensor), and number of batches (int).
      X_batch and Y_batch have of the shape [batch_size, maxlen].
    '''
    # Load data
    X = load_train_data() # (196947, 1000) int64

    # Create Queues
    x_q, = tf.train.slice_input_producer([tf.convert_to_tensor(X, tf.int32)]) # (1000,) int32
    
    # Lstrip zeros
    zeros = tf.equal(x_q, tf.zeros_like(x_q)).sg_int().sg_sum()
    x_q = x_q[zeros:] 
    
    # Bi-Padding
    x_q = tf.concat(0, [tf.zeros([Hyperparams.seqlen], tf.int32), # 50 zero-prepadding
                        x_q, 
                        [3], # 1 space
                        tf.zeros([Hyperparams.seqlen-1], tf.int32)]) # 49 zero-postpadding

    # Random crop 
    x_q = tf.random_crop(x_q, [2 * Hyperparams.seqlen]) # (2 * 50,) int64

    # Split into x and y
    x_q, y_q = x_q[:Hyperparams.seqlen], x_q[Hyperparams.seqlen:] # (50,) int64, (50,) int64

    # Label processing
    y_q = label_process(y_q) # (50,) int32
    y_q.set_shape((50,))
#     
    # create batch queues
    x, y = tf.train.shuffle_batch([x_q, y_q],
                              num_threads=32,
                              batch_size=Hyperparams.batch_size, 
                              capacity=Hyperparams.batch_size*64,
                              min_after_dequeue=Hyperparams.batch_size*32, 
                              allow_smaller_final_batch=False)  # (16, 100)
    
    num_batch = len(X) // Hyperparams.batch_size

    return x, y, num_batch # (64, 50) int32, (64, 50) int32, ()

class ModelGraph():
    '''Builds a model graph'''
    def __init__(self, mode="train"):
        '''
        Args:
          mode: A string. Either "train" , "val", or "test"
        '''
        self.char2idx, self.idx2char = load_vocab()
        
        if mode == "train":
            self.x, self.y, self.num_batch = get_batch_data() 
            self.y_src = tf.concat(1, [tf.zeros((Hyperparams.batch_size, 1), tf.int32), self.y[:, :-1]])
        else:
            self.x = tf.placeholder(tf.int32, [None, Hyperparams.seqlen])
            self.y_src = tf.placeholder(tf.int32, [None, Hyperparams.seqlen])
        
        # make embedding matrix for input characters
        self.emb_x = tf.sg_emb(name='emb_x', voca_size=len(self.char2idx), dim=Hyperparams.embed_dim)
        
        self.enc = self.x.sg_lookup(emb=self.emb_x)
        
        with tf.sg_context(size=5, act='relu', bn=True):
            for _ in range(20):
                dim = self.enc.get_shape().as_list()[-1]
                self.enc += self.enc.sg_conv1d(dim=dim) # (64, 50, 300) float32
        
        self.enc = tf.reduce_mean(self.enc, reduction_indices=[1], keep_dims=True) # (64, 1, 300) float32
        self.enc = tf.tile(self.enc, [1, 50, 1]) # (64, 50, 300) <dtype: 'float32'>
        self.enc = self.enc.sg_concat(target=self.y_src.sg_lookup(emb=self.emb_x))
                 
        self.dec = self.enc
        with tf.sg_context(size=5, act='relu', bn=True):
            for _ in range(20):
                dim = self.dec.get_shape().as_list()[-1]
                self._dec = tf.pad(self.dec, [[0, 0], [4, 0], [0, 0]])  # zero prepadding
                self.dec += self._dec.sg_conv1d(dim=dim, pad='VALID') 
        
        self.logits = self.dec.sg_conv1d(size=1, dim=len(self.char2idx), act='linear', bn=False) # (64, 50, 5072) float32

        if mode == "train":
            self.ce = self.logits.sg_ce(target=self.y, mask=True, one_hot=False)
            self.istarget = tf.not_equal(self.y, tf.zeros_like(self.y)).sg_float() # (64, 50) float32
            self.reduced_loss = (self.ce.sg_sum()) / (self.istarget.sg_sum() + 1e-5)
            tf.sg_summary_loss(self.reduced_loss, "reduced_loss")
            
def train():
    g = ModelGraph()
    print("Graph loaded!")

    tf.sg_train(lr=0.0001, lr_reset=True, log_interval=10, loss=g.reduced_loss, eval_metric=[], max_ep=2000, 
                save_dir='asset/train', early_stop=False, max_keep=10, ep_size=g.num_batch)
     
if __name__ == '__main__':
    train(); print("Done")
