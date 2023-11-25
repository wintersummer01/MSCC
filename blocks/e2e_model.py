import tensorflow as tf
import tensorflow_compression as tfc
from keras import Model
import sionna as sn

from blocks.latentRepresentation_g import g_a, g_s
from utils.config import config

class NTC_simple(Model):
    def __init__(self, shape=(32, 32, 3)):
        super().__init__()
        # Trainable Modules
        self.latent_analysis = g_a(shape)
        self.latent_synthesis = g_s(shape)

    def call(self, x):
        # Analysis Function
        y = self.latent_analysis(x)
        norm = tf.reduce_sum(y**2)**0.5
        y /= norm
        
        # Synthesis Function (latent y)
        x_hat = self.latent_synthesis(y)
        
        return x_hat