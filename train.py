# -*- coding: utf-8 -*-
'''
Tokenizes English sentences using neural networks
Nov., 2016. Kyubyong.
'''

from prepro import Hyperparams, load_data, load_charmaps
import sugartensor as tf

def get_batch_data():
    '''Makes batch queues from the data.
    
    Args:
      mode: A string. Either 'train', 'val', or 'test' 
    Returns:
      A Tuple of X_batch (Tensor), Y_batch (Tensor), and number of batches (int).
      X_batch and Y_batch have of the shape [batch_size, maxlen].
    '''
    # Load data
    X = load_data() # (9207, 1000)

    # Create Queues
    x_q, = tf.train.slice_input_producer([tf.convert_to_tensor(X, tf.int32)]) # (1000,)
    
    # Lstrip zeros
    zeros = tf.equal(x_q, tf.zeros_like(x_q)).sg_int().sg_sum()
    x_q = x_q[zeros:] 
    
    # Initial Padding
    x_q = tf.concat(0, [tf.zeros([Hyperparams.seqlen-1,], tf.int32), # 99 zero-padding
                        tf.ones([1,], tf.int32),  # 1: space
                        x_q]) # [0, 0, 0, ..., 1, ...] 99 zero-paddings and 1 space and real numbers
    
    # Random crop
    x_q = crop(x_q, Hyperparams.seqlen+1) # (101,) Why? the last one will be y.
    x_q.set_shape(Hyperparams.seqlen+1)
#     
    # create batch queues
    x = tf.train.shuffle_batch([x_q],
                              num_threads=32,
                              batch_size=Hyperparams.batch_size, 
                              capacity=Hyperparams.batch_size*64,
                              min_after_dequeue=Hyperparams.batch_size*32, 
                              allow_smaller_final_batch=False)  # (16, 100)
    
    return x

def crop(x, seqlen):
    '''Returns random cropped tensor `y` of `x`.
    To avoid `y` starts with broken word piece, we replace the elements before the 
      the first appearing 1 (space) with zeros.
    
    For example,
    
    ```
    import tensorflow as tf
    x = tf.constant([3, 5, 2, 1, 5, 6, 7, 1, 3])
    seqlen = 5
    z = crop(x, seqlen)
    with tf.Session() as sess:
        print z.eval()
        => [0 0 1 5 6]
    ```
       
    Args:
      x: A 1-D `Tensor`.
      seqlen: A 1-D `Tensor`. Seqlen of the returned tensor.
    
    Returns: 
      A 1-D `Tensor`. Has the size of seqlen.
    
    '''
    x = tf.random_crop(x, [seqlen]) # (100,)
    first_ind_of_1 = tf.where(tf.equal(x, 1))[0] # 1 means space.
    
    zero_padding = tf.zeros(tf.to_int32(first_ind_of_1), tf.int32)
    x_back = tf.slice(x, first_ind_of_1, [-1]) # -1 means "all the remaining part"
    out = tf.concat(0, (zero_padding, x_back))
    
    return out
    
    
# residual block
@tf.sg_sugar_func
def sg_res_block(tensor, opt):
    # default rate
    opt += tf.sg_opt(size=5, rate=1, causal=False)

    # input dimension
    in_dim = tensor.get_shape().as_list()[-1]

    # reduce dimension
    input_ = (tensor
              .sg_bypass(act='relu', bn=(not opt.causal), ln=opt.causal)
              .sg_conv1d(size=1, dim=in_dim/2, act='relu', bn=(not opt.causal), ln=opt.causal))

    # 1xk conv dilated
    out = input_.sg_aconv1d(size=opt.size, rate=opt.rate, causal=opt.causal, act='relu', bn=(not opt.causal), ln=opt.causal)

    # dimension recover and residual connection
    out = out.sg_conv1d(size=1, dim=in_dim) + tensor

    return out

# inject residual multiplicative block
tf.sg_inject_func(sg_res_block)

class ModelGraph():
    '''Builds a model graph'''
    def __init__(self, is_train=True):
        '''
        Args:
          mode: A string. Either "train" , "val", or "test"
        '''
        if is_train:
            self.x = get_batch_data() # (16, 101)
            self.x, self.y = self.x[:, :Hyperparams.seqlen], self.x[:, Hyperparams.seqlen] #(16, 100) (16,)
        else:
            self.x = tf.placeholder(tf.int32, [None, Hyperparams.seqlen])
        
        self.char2idx, self.idx2char = load_charmaps()
        
        # make embedding matrix for input characters
        self.emb_x = tf.sg_emb(name='emb_x', voca_size=len(self.char2idx), dim=Hyperparams.embed_dim)
        self.enc = self.x.sg_lookup(emb=self.emb_x).sg_float()
        
        # loop dilated causal conv block
        with tf.sg_context(size=5, causal=False):
            for _ in range(5):
                self.enc = (self.enc
                       .sg_res_block(rate=1)
                       .sg_res_block(rate=2)
                       .sg_res_block(rate=4)
                       .sg_res_block(rate=8)
                       .sg_res_block(rate=16))
        # final fully convolution layer for softmax
        self.logits = self.enc.sg_conv1d(size=1, dim=len(self.char2idx))#.sg_sum(dims=1) # (16, 99, 70)
#         self.nonzeros = tf.not_equal(self.x, tf.zeros_like(self.x)).sg_float().sg_sum(dims=1).sg_expand_dims() # (16, 1)
#         self.logits = self.logits / self.nonzeros # (16, 70)
        self.logits = tf.reduce_mean(self.logits, reduction_indices=[1], keep_dims=False) # (16, 70)
#         print self.logits.get_shape()
        
        if is_train:
            # Cross entropy loss
            self.ce = self.logits.sg_ce(target=self.y, mask=False, one_hot=False)
            
def train():
    g = ModelGraph()
    print "Graph loaded!"

    tf.sg_train(lr_reset=True, log_interval=10, loss=g.ce, eval_metric=[], max_ep=100, 
                save_dir='asset/train', early_stop=False, max_keep=10)
     
if __name__ == '__main__':
    train(); print "Done"
