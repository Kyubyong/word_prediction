# -*- coding: utf-8 -*-
'''
Tokenizes English sentences using neural networks
Nov., 2016. Kyubyong.
'''

from prepro import Hyperparams, load_train_data, load_vocab
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
    X = load_train_data(mode=mode) # (9207, 1000)

    # Create Queues
    x_q, = tf.train.slice_input_producer([tf.convert_to_tensor(X, tf.int64)]) # (1000,)
    
    # Lstrip zeros
    zeros = tf.equal(x_q, tf.zeros_like(x_q)).sg_int().sg_sum()
    x_q = x_q[zeros:] 
    
    # Initial Padding
    x_q = tf.concat(0, [tf.zeros([Hyperparams.ctxlen], tf.int64), # 50 zero-padding
                        x_q]) # [0, 0, 0, ..., 0, ...] 50 zero-paddings and real numbers
    
    # Random crop
    x_q = tf.random_crop(x_q, [Hyperparams.ctxlen + Hyperparams.predlen]) # (50+10,)
#     x_q.set_shape(Hyperparams.ctxlen + Hyperparams.predlen)
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
        self.char2idx, self.idx2char = load_vocab()
        
        if is_train:
            self.x, self.num_batch = get_batch_data() # (64, 60) int64
            _x, _y = self.x[:, :Hyperparams.ctxlen], self.x[:, Hyperparams.ctxlen:] #(64, 50) (64, 10)
            self.x = tf.concat(1, [_x, tf.ones_like(_y) * len(self.char2idx)]) # (64, 60) int64
            self.y = tf.concat(1, [tf.zeros_like(_x), _y]) # (64, 60) int64

            self.x_val, _ = get_batch_data(mode="val")
            _x_val, _y_val = self.x_val[:, :Hyperparams.ctxlen], self.x_val[:, Hyperparams.ctxlen:] #(64, 50) (64, 10)
            self.x_val = tf.concat(1, [_x_val, tf.ones_like(_y_val) * len(self.char2idx)]) # (64, 60)
            self.y_val = tf.concat(1, [tf.zeros_like(_x_val), _y_val]) # (64, 60)            
        else:
            self.x = tf.placeholder(tf.int32, [None, Hyperparams.ctxlen+Hyperparams.predlen])
        
        # make embedding matrix for input characters
        self.emb_x = tf.sg_emb(name='emb_x', voca_size=len(self.char2idx)+1, dim=Hyperparams.embed_dim)
        self.enc = self.x.sg_lookup(emb=self.emb_x)
        
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
#         self.logits = tf.reduce_mean(self.logits, reduction_indices=[1], keep_dims=False) # (16, 70)
#         print self.logits.get_shape()
        
        if is_train:
            # Cross entropy loss
            self.ce = self.logits.sg_ce(target=self.y, mask=True, one_hot=False)
            self.istarget = tf.not_equal(self.y, tf.zeros_like(self.y)).sg_float()
            self.reduced_loss = self.ce.sg_sum() / self.istarget.sg_sum() # ()
            tf.sg_summary_loss(self.reduced_loss, "reduced_loss")

            # Validation check
            self.preds = (self.logits.sg_reuse(input=self.x_val)).sg_argmax()
            self.hits = (tf.equal(self.preds, self.y_val)).sg_float()
            self.istarget_ = (tf.not_equal(self.y_val, tf.zeros_like(self.y_val))).sg_float()
            self.acc = (self.hits * self.istarget_).sg_sum() / (self.istarget_.sg_sum() + tf.sg_eps) # ()
                        
def train():
    g = ModelGraph()
    print "Graph loaded!"

    tf.sg_train(lr_reset=True, log_interval=10, loss=g.reduced_loss, eval_metric=[g.acc], max_ep=100, 
                save_dir='asset/train', early_stop=False, max_keep=10, ep_size=g.num_batch)
     
if __name__ == '__main__':
    train(); print "Done"
