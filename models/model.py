import tensorflow as tf
import tensorflow_compression as tfc
from keras import layers, Model
import sionna as sn
from models.basicBlocks import encoding_stage, decoding_stage


class MSCC_encoder(layers.Layer):
    def __init__(self, img_shape, channel_dim, ps, num_blocks, num_heads, mlp_ratio):
        super().__init__()
        H, W, _ = img_shape
        self.stage1 = encoding_stage(img_shape, channel_dim, ps, num_blocks[0], 
                                     num_heads, mlp_ratio)
        hidden_shape = (int(H/ps), int(W/ps), channel_dim)
        self.stage2 = encoding_stage(hidden_shape, channel_dim, ps, num_blocks[1], 
                                     num_heads, mlp_ratio)
        
    def __call__(self, x):
        x = self.stage1(x)
        y = self.stage2(x)
        
        norm = tf.reduce_sum(y**2)**0.5
        y /= norm
        
        y = tf.reshape(y, [2, -1])
        symbol = tf.complex(y[0], y[1])
        return symbol


class MSCC_decoder(layers.Layer):
    def __init__(self, img_shape, channel_dim, ps, num_blocks, num_heads, mlp_ratio):
        super().__init__()
        self.MIM
        H, W, C = img_shape
        self.symbol_shape = (int(H/(ps**2)), int(W/(ps**2)), channel_dim)
        self.stage2 = decoding_stage(self.symbol_shape, channel_dim, ps, num_blocks[1], 
                                     num_heads, mlp_ratio)
        hidden_shape = (int(H/ps), int(W/ps), channel_dim)
        self.stage1 = decoding_stage(hidden_shape, channel_dim, ps, num_blocks[0], 
                                     num_heads, mlp_ratio)
        self.recovery = tfc.layers.SignalConv2D(C, 3, padding='same_zeros')
        
    def __call__(self, symbol_hat):
        y_hat = tf.stack([tf.math.real(symbol_hat), tf.math.imag(symbol_hat)])
        y_hat = tf.reshape(y_hat, self.symbol_shape)
        
        y_hat = self.stage2(y_hat)
        y_hat = self.stage1(y_hat)
        x_hat = self.recovery(y_hat)
        return x_hat
     

class MSCC_end2end(Model):
    def __init__(self, **model_kwargs):
        super().__init__()
        self.encoder = MSCC_encoder(**model_kwargs)
        self.channel = sn.channel.AWGN()
        self.decoder = MSCC_decoder(**model_kwargs)
        
    def call(self, x, no):
        s = self.encoder(x)
        s_hat = self.channel([s, no])
        x_hat = self.decoder(s_hat)
        return x_hat