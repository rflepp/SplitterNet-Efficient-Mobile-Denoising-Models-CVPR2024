import tensorflow as tf
from tensorflow.keras import layers, Model

NUM_LEVELS = 4


def reflect_pad(x):
    return tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')


def split_channels(x):
    """Split a tensor into two halves along the channel axis."""
    half = x.shape[3] // 2
    return x[..., :half], x[..., half:2 * half]


def down_blk(x, filters):
    x = layers.Conv2D(filters, kernel_size=3, strides=2, padding='valid')(reflect_pad(x))
    return layers.LeakyReLU()(x)


def decoder_blk(branch_1, branch_2, skip, filters):
    x = tf.concat([branch_1, branch_2], axis=3)
    x = layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same')(x)
    return layers.LeakyReLU()(x) + skip


def mid_blk(inputs, filters):
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='valid')(reflect_pad(inputs))
    x = layers.LeakyReLU()(x)
    residual = simpl_chan_att(x, filters) + inputs
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='valid')(reflect_pad(residual))
    x = layers.LeakyReLU()(x)
    return spatial_attention(x) + residual


def spatial_attention(inputs):
    avg_pool = tf.reduce_mean(inputs, axis=3, keepdims=True)
    max_pool = tf.reduce_max(inputs, axis=3, keepdims=True)
    x = reflect_pad(tf.concat([avg_pool, max_pool], axis=3))
    attention = layers.Conv2D(1, kernel_size=3, strides=1, padding='valid', activation='sigmoid')(x)
    return layers.Multiply()([inputs, attention])


def simpl_chan_att(inputs, channels):
    x = layers.GlobalAveragePooling2D()(inputs)
    x = tf.reshape(x, shape=(-1, 1, 1, channels))
    return inputs * layers.Conv2D(channels, kernel_size=1)(x)


def DYNUnet(input_shape=(None, None, 3), num_filters=32):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same')(inputs)

    # Encoder: at every level each branch is split along the channels and both
    # halves are downsampled independently, doubling the number of branches.
    skips = []
    branches = [x]
    for _ in range(NUM_LEVELS):
        skips.append(branches)
        branches = [down_blk(half, num_filters) for b in branches for half in split_channels(b)]

    branches = [mid_blk(b, num_filters) for b in branches]

    # Decoder: fuse neighbouring branch pairs and add the matching encoder skip.
    for level_skips in reversed(skips):
        branches = [decoder_blk(branches[2 * i], branches[2 * i + 1], skip, num_filters)
                    for i, skip in enumerate(level_skips)]

    out = layers.Conv2D(3, kernel_size=3, activation=None, padding='same')(branches[0])
    return Model(inputs=[inputs], outputs=[inputs + out])
