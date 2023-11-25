import tensorflow as tf
import tensorflow_compression as tfc
from keras import layers
from blocks.basicBlock import transformer_block

class g_a(layers.Layer):
    def __init__(self, shape, dim=256, ws=16, num_blocks=[2, 6]):
        super().__init__()
        self.H, self.W, self.C = shape
        self.dim = dim
        self.ws = ws
        self.ps = int(self.H/self.ws)
        transformer_shape = [(self.ws**2, self.dim),
                             (int(self.ws/2)**2, self.dim)]
        # For Stage 1
        self.patch_embedding = layers.Dense(self.dim)
        self.layer1 = []
        for _ in range(num_blocks[0]):
            self.layer1.append(transformer_block(transformer_shape[0]))
        # For Stage 2
        self.patch_merge = layers.Dense(self.dim)
        self.layer2 = []
        for _ in range(num_blocks[1]):
            self.layer2.append(transformer_block(transformer_shape[1]))

    def __call__(self, x):

        # Stage1
        # Patch Partition       
        patch = tf.reshape(x, [-1, self.ws, self.ps, self.ws, self.ps*self.C])
        patch = tf.transpose(patch, [0, 1, 3, 2, 4])
        patch = tf.reshape(patch, [-1, self.ws**2, (self.ps**2)*self.C])
        token = self.patch_embedding(patch)
        # Transformer Blocks
        for block in self.layer1:
            token = block(token)
        
        # Stage 2
        # Patch Merging
        hidden_patch = tf.reshape(token, [-1, int(self.H/4), 2, int(self.W/4), 2*self.dim])
        hidden_patch = tf.transpose(hidden_patch, [0, 1, 3, 2, 4])
        hidden_patch = tf.reshape(hidden_patch, [-1, int((self.H/4)*(self.W/4)), self.dim*4])
        y = self.patch_merge(hidden_patch)
        # Transformer Blocks
        for block in self.layer2:
            y = block(y)

        # Shape rebuilding
        y = tf.reshape(y, [-1, int(self.H/4), int(self.W/4), self.dim])

        return y
    

class g_s(layers.Layer):
    def __init__(self, shape, dim=256, ws=16, num_blocks=[2, 6]):
        super().__init__()
        self.H, self.W, self.C = shape
        self.dim = dim
        self.ws = ws
        self.ps = int(self.H/self.ws)
        transformer_shape = [(self.ws**2, self.dim),
                             (int(self.ws/2)**2, self.dim)]
        # For Stage 2
        self.layer2 = []
        for _ in range(num_blocks[1]):
            self.layer2.append(transformer_block(transformer_shape[1]))
        self.patch_division2 = layers.Dense(self.dim*4)
        # For Stage 1
        self.layer1 = []
        for _ in range(num_blocks[0]):
            self.layer1.append(transformer_block(transformer_shape[0]))
        self.patch_division1 = layers.Dense(self.dim*4)
        # For output CNN
        self.recovery = tfc.layers.SignalConv2D(3, 3, padding='same_zeros')
        

    def __call__(self, y_bar):
        # Shape Building
        hidden_patch = tf.reshape(y_bar, [-1, int((self.H/4)*(self.W/4)), self.dim])

        # Stage 2
        # Transformer Blocks
        for block in self.layer2:
            hidden_patch = block(hidden_patch)
        # Patch Division
        hidden_patch = self.patch_division2(hidden_patch)
        hidden_token = tf.reshape(hidden_patch, [-1, int(self.H/4), int(self.W/4), 2, 2*self.dim])
        hidden_token = tf.transpose(hidden_token, [0, 1, 3, 2, 4])
        patch = tf.reshape(hidden_token, [-1, int((self.H/2)*(self.W/2)), self.dim])
        
        # Stage 1
        # Transformer Blocks
        for block in self.layer1:
            patch = block(patch)
        # Patch Division
        patch = self.patch_division1(patch)
        token = tf.reshape(patch, [-1, int(self.H/2), int(self.W/2), self.ps, self.ps*self.dim])
        token = tf.transpose(token, [0, 1, 3, 2, 4])
        token = tf.reshape(token, [-1, self.H, self.W, self.dim])
        
        # recovery CNN
        x_bar = self.recovery(token)
        
        return x_bar