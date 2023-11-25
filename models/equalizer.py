import tensorflow as tf
from keras import layers
from models.basicBlocks import transformer_block

class dnn_equalizer(layers.Layer):
    def __init__(self, decoder_shape, coder_kwargs):
        super().__init__()
        _, self.L, self.C = decoder_shape
        num_heads = coder_kwargs['num_heads']
        mlp_ratio = coder_kwargs['mlp_ratio']
        self.num_symbols_per_batch = self.L*self.C//2
        self.block = transformer_block(self.L, self.C*2, num_heads, mlp_ratio)
        self.projection = layers.Dense(self.C)
    
    def __call__(self, s_hat, h):
        s_hat = tf.reshape(s_hat, [-1, self.num_symbols_per_batch, 1])
        h = tf.reshape(h, [-1, self.num_symbols_per_batch, 1])
        
        y_hat = tf.concat([tf.math.real(s_hat), tf.math.imag(s_hat), tf.math.real(h), tf.math.imag(h)], axis=2)
        y_hat = tf.reshape(y_hat, [-1, self.L, self.C*2])
        y_hat = self.block(y_hat)
        y_hat = self.projection(y_hat)
        
        return y_hat
        
        