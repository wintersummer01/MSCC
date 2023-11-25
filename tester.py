#%%

from utils.dataset import loadCifarDataset

A, B = loadCifarDataset(64, 64, '/workspace/dataset/cifar-10-batches-py/')

k = iter(A)
l = k.get_next()
print(l[0])
print(l[1])

j = 0
for i in A:
    j += i[0].shape[0]
print(j)

# %%
import tensorflow as tf
x = tf.zeros([3, 2], dtype=tf.complex64)

layer = tf.keras.layers.Lambda(lambda x: tf.stack([tf.math.real(x[0]), \
    tf.math.imag(x[0]), tf.math.real(x[0]), tf.math.imag(x[0])]))
y = layer([x, x])
print(y)

# %%
import tensorflow as tf
from models.model import MSCC_end2end
from utils.config import config

a = tf.zeros((1, 32, 32, 3))
m = MSCC_end2end(
        snr_db=      config.TRAIN_SNR,
        img_shape=   config.IMAGE_SHAPE,
        channel_dim= config.CHANNEL_DIMENSION,
        ps=          config.PATCH_MERGE_SIZE,
        num_blocks=  config.NUM_BLOCKS_IN_LAYER,
        num_heads=   config.NUM_MHSA_HEADS,
        mlp_ratio=   config.MLP_RATIO
    )
m(a)
# %%
import tensorflow as tf
a = tf.reshape(tf.range(30, dtype=tf.float32), [3, 5, 2])
b = tf.reshape(tf.range(30, dtype=tf.float32)+0.1, [3, 5, 2])
print(a)
print(b)
a = tf.complex(a[:,:,0], a[:,:,1])
b = tf.complex(b[:,:,0], b[:,:,1])
print(a, b)
a = tf.reshape(a, [-1, 1])
b = tf.reshape(b, [-1, 1])
a = tf.reshape(a, [3, 5, 1])
b = tf.reshape(b, [3, 5, 1])
c = tf.concat([tf.math.real(a), tf.math.imag(a), tf.math.real(b), tf.math.imag(b)], axis=2)
c = tf.reshape(c, [-1, 2, 10])
# a = tf.stack([tf.math.real(a), tf.math.imag(a), tf.math.real(b), tf.math.imag(b)])
# print(a)
# a = tf.reshape(a, [2, 3, 10])
# print(a)
# a = tf.transpose(a, [1, 0, 2])
# print(a)
# a = tf.reshape(a, [-1, 20])
print(c)
# %%
import sionna as sn
import tensorflow as tf

gen = sn.channel.GenerateFlatFadingChannel(1, 1)
cha = sn.channel.ApplyFlatFadingChannel(add_awgn=False)

h = gen(10)

# %%
# print(h)
x = tf.complex(tf.ones([10, 1]), tf.zeros([10, 1]))
y = cha([x, h])
# print(y)
# print(h)
# print(x*tf.squeeze(h, axis=2))
print(sn.mimo.zf_equalizer(y, h, 100*tf.ones([1, 1], dtype=tf.complex64))[1])
# print(y/tf.squeeze(h, axis=2))



# %%
from utils.config import config
from utils.dataset import loadCifarDataset

test_ds = loadCifarDataset(
    None,
    config.TEST_BATCH_SIZE,
    config.DATA_ROOT,
    type='test',
    mini=True
)

img = iter(test_ds[0]).get_next()[0][11,:]
print(img.shape)
# %%
