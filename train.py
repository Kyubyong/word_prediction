# -*- coding: utf-8 -*-
'''
Tokenizes English sentences using neural networks
Nov., 2016. Kyubyong.
'''

from __prepro import Hyperparams, load_data, load_charmaps
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
    X = load_data() # (11887, 11648)

    # Create Queues
    x_q, = tf.train.slice_input_producer([tf.convert_to_tensor(X, tf.int32)]) # (11747,)
    
    # Lstrip zeros
    zeros = tf.equal(x_q, tf.zeros_like(x_q)).sg_int().sg_sum()
    x_q = x_q[zeros:]
    
    # Append 99 zeros.
    x_q = tf.concat(0, [tf.zeros([Hyperparams.seqlen-1,], tf.int32), x_q])
    
    # Random crop
    x_q = tf.random_crop(x_q, [Hyperparams.seqlen]) # (100,)
#     
    # create batch queues
    x = tf.train.shuffle_batch([x_q],
                              num_threads=32,
                              batch_size=Hyperparams.batch_size, 
                              capacity=Hyperparams.batch_size*64,
                              min_after_dequeue=Hyperparams.batch_size*32, 
                              allow_smaller_final_batch=False)  # (16, 100)
    
    return x

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
            self.x = get_batch_data() # (16, 11747)
            self.x, self.y = self.x[:, :Hyperparams.seqlen-1], self.x[:, Hyperparams.seqlen-1] # (16, 99) (16,)
        else:
            self.x = tf.placeholder(tf.int32, [None, Hyperparams.seqlen-1])
        
        
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
#             self.nonzeros = tf.not_equal(self.y, tf.zeros_like(self.y)).sg_float()
#             self.reduced_loss = (self.ce * self.nonzeros).sg_sum() / ( self.nonzeros.sg_sum() + tf.sg_eps )
            
def train():
    g = ModelGraph()
    print "Graph loaded!"

    tf.sg_train(lr=0.0001, lr_reset=True, log_interval=10, loss=g.ce, eval_metric=[], max_ep=100, 
                save_dir='asset/train', early_stop=False, max_keep=10)
     
if __name__ == '__main__':
    train(); print "Done"
