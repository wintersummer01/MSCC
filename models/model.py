import tensorflow as tf
import tensorflow_compression as tfc
from keras import layers, Model
import sionna as sn
from models.basicBlocks import encoding_stage, decoding_stage
from models.equalizer import dnn_equalizer

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
        return y


class MSCC_decoder(layers.Layer):
    def __init__(self, img_shape, channel_dim, ps, num_blocks, num_heads, mlp_ratio):
        super().__init__()
        H, W, C = img_shape
        self.patch_shape = (-1, H, W, channel_dim)
        symbol_shape = (H//(ps**2), W//(ps**2), channel_dim)
        self.stage2 = decoding_stage(symbol_shape, channel_dim, ps, num_blocks[1], 
                                     num_heads, mlp_ratio)
        hidden_shape = (H//ps, W//ps, channel_dim)
        self.stage1 = decoding_stage(hidden_shape, channel_dim, ps, num_blocks[0], 
                                     num_heads, mlp_ratio)
        self.recovery = tfc.layers.SignalConv2D(C, 3, padding='same_zeros')
        
    def __call__(self, y_hat):
        y_hat = self.stage2(y_hat)
        y_hat = self.stage1(y_hat)
        y_hat = tf.reshape(y_hat, self.patch_shape)
        x_hat = self.recovery(y_hat)
        return x_hat


class MSCC_end2end(Model):
    def __init__(self, num_tx, num_rx, snr_db, channel_type, equalizer_type, **coder_kwargs):
        super().__init__()
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.no = sn.utils.ebnodb2no(snr_db, num_bits_per_symbol=1, coderate=1.0)
        # Enc / Dec
        self.encoder = MSCC_encoder(**coder_kwargs)
        self.decoder = MSCC_decoder(**coder_kwargs)
        
        # Channel
        if channel_type == 'flat':
            self.channel = sn.channel.FlatFadingChannel(self.num_tx, self.num_rx,
                                                         add_awgn=True, return_channel=True)
        elif channel_type == 'awgn':
            self.awgn_channel = sn.channel.AWGN()
            def awgn(inp):
                s_hat = self.awgn_channel(inp)
                return s_hat, 0
            self.channel = awgn
            
        # Equalizer
        H, W, _ = coder_kwargs['img_shape']
        self.dim = coder_kwargs['channel_dim']
        self.decoder_shape = (-1, (H*W)//(coder_kwargs['ps']**4), self.dim)
        
        if equalizer_type is None:
            def no_equalizing(s_hat, h):
                s_hat = tf.reshape(s_hat, [-1, self.decoder_shape[1]*self.dim//2, 1])
                y_hat = tf.concat([tf.math.real(s_hat), tf.math.imag(s_hat)], axis=2)
                y_hat = tf.reshape(y_hat, self.decoder_shape)
                return y_hat
            self.equalizer = no_equalizing
        elif equalizer_type == 'zf':
            def zero_forcing(s_hat, h):
                s_hat, _ = sn.mimo.zf_equalizer(s_hat, h, tf.zeros([self.num_tx, self.num_rx], dtype=tf.complex64))
                s_hat = tf.reshape(s_hat, [-1, self.decoder_shape[1]*self.dim//2, 1])
                y_hat = tf.concat([tf.math.real(s_hat), tf.math.imag(s_hat)], axis=2)
                return tf.reshape(y_hat, self.decoder_shape)
            self.equalizer = zero_forcing
        elif equalizer_type == 'dnn':
            self.equalizer = dnn_equalizer(self.decoder_shape, coder_kwargs)
        
        
    def call(self, x, no=None):
        # Encoder
        y = self.encoder(x)
        # Transmitter (Precoder)
        norm = tf.reduce_mean(y**2)**0.5
        y /= norm
        
        y = tf.reshape(y, [-1, self.decoder_shape[1]*self.dim//2, 2])
        s = tf.complex(y[:,:,0], y[:,:,1])
        s = tf.reshape(s, [-1, self.num_tx])
        
        # Channel
        if no is None:
            no = self.no
        s_hat, h = self.channel([s, no])
        
        # Receiver (Equalizer)
        y_hat = self.equalizer(s_hat, h)
        y_hat *= norm
        # Decoder
        x_hat = self.decoder(y_hat)
        return x_hat
    

class MSCC_easy(Model):
    def __init__(self, **coder_kwargs):
        super().__init__()
        self.encoder = MSCC_encoder(**coder_kwargs)
        self.decoder = MSCC_decoder(**coder_kwargs)
        
    def call(self, x):
        y = self.encoder(x)
        norm = tf.reduce_sum(y**2)**0.5
        y /= norm
        y *= norm
        x_hat = self.decoder(y)
        return x_hat
    
    
if __name__ == "__main__":
    from utils.config import config

    a = tf.ones((3, 32, 32, 3))
    m = MSCC_end2end(
            snr_db=      config.TRAIN_SNR,
            num_tx=      1,
            num_rx=      1,
            img_shape=   config.IMAGE_SHAPE,
            channel_dim= config.CHANNEL_DIMENSION,
            ps=          config.PATCH_MERGE_SIZE,
            num_blocks=  config.NUM_BLOCKS_IN_LAYER,
            num_heads=   config.NUM_MHSA_HEADS,
            mlp_ratio=   config.MLP_RATIO
        )
    m(a)
