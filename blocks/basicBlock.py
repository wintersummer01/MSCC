import tensorflow as tf
from keras import layers

# Learnable relative positional bias
class positional_bias(layers.Layer):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.values = tf.Variable(tf.range((2*window-1)**2, dtype=tf.float32), trainable=True, name='positional_encoding')
        position = tf.reshape(tf.meshgrid(tf.range(window), tf.range(window)), [2, -1])
        position_map = position[:,:,None] - position[:,None,:] + window-1
        self.position_index = tf.reshape(position_map[0] + position_map[1]*(2*window-1), [-1])

    def __call__(self, input):
        relative_embedding = tf.reshape(tf.gather(self.values, self.position_index), [self.window**2, -1])
        return relative_embedding + input
    

# Basic Transformer Block
# Only Considering about CIFAR10 Dataset 
class transformer_block(layers.Layer):
    def __init__(self, shape, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.L, self.C = shape
        self.num_heads = num_heads
        # For Norm
        self.norm = layers.LayerNormalization(-1, epsilon=1e-5)
        # For MHSA
        self.to_qkv = layers.Dense(self.C*3)
        self.bias = positional_bias(int(self.L ** 0.5))
        self.softmax = layers.Softmax()
        self.projection = layers.Dense(self.C)
        # For MLP
        self.fc1 = layers.Dense(int(self.C*mlp_ratio))
        self.fc2 = layers.Dense(self.C)
        self.activation = tf.keras.activations.gelu


    # input dimension = B * L * C
    # B: Batch size
    # L: number of tokens (h/2 * w/2 = window_size**2 = 16**2)
    # C: token dimension (256)
    def __call__(self, x):
        # LayerNorm
        norm_x = self.norm(x)

        # Multi-Head Self Attention
        QKV = tf.reshape(self.to_qkv(norm_x), [-1, self.L, 3, self.num_heads, int(self.C/self.num_heads)])
        QKV = tf.transpose(QKV, [2, 0, 3, 1, 4])
        Q, K, V = QKV[0], QKV[1], QKV[2]
        attention = tf.linalg.matmul(Q, K, transpose_b=True)/tf.math.sqrt(self.C/self.num_heads)
        raw_mhsa = tf.linalg.matmul(self.softmax(self.bias(attention)), V)
        mhsa = self.projection(tf.reshape(tf.transpose(raw_mhsa, [0, 2, 1, 3]), [-1, self.L, self.C]))
        
        # Add
        output1 = x + mhsa
        
        # LayerNorm
        norm_output1 = self.norm(output1)
        
        # MLP
        mlp = self.fc1(norm_output1)
        mlp = self.activation(mlp)
        mlp = self.fc2(mlp)
        ## mlp = self.activation(mlp) # Doesn't need another activation?
        
        # Add
        output2 = output1 + mlp
        
        return output2
        