import tensorflow as tf
from tensorflow.keras import layers
import logging
from tensorflow import keras

logging.basicConfig(level=logging.DEBUG)

def encoder_blk(input, filters):
    net = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    net = layers.Conv2D(filters, kernel_size=3, strides=1, padding='valid')(net)
    net = tf.keras.layers.LeakyReLU()(net)
    return net

def decoder_blk(net, filters):
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    net = layers.Conv2D(filters, kernel_size=3, strides=1, padding="valid")(net)
    net = tf.keras.layers.LeakyReLU()(net)

    split_tensors = split_using_slice(net)

    split_tensor_1 = split_tensors[0]
    split_tensor_2 = split_tensors[1]

    split_tensor_1 = tf.pad(split_tensor_1, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    split_tensor_1 = layers.Conv2D(filters, kernel_size=3, strides=1, padding='valid')(split_tensor_1)
    split_tensor_1 = tf.keras.layers.LeakyReLU()(split_tensor_1)

    split_tensor_2 = tf.pad(split_tensor_2, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    split_tensor_2 = layers.Conv2D(filters, kernel_size=3, strides=1, padding='valid')(split_tensor_2)
    split_tensor_2 = tf.keras.layers.LeakyReLU()(split_tensor_2)

    concatenated_tensor = tf.concat([split_tensor_1, split_tensor_2], axis=3)

    return concatenated_tensor

def split_using_slice(tensor):
    # Slice the first half of the last dimension (channels)
    slice1 = tf.slice(tensor, [0, 0, 0, 0], [-1, -1, -1, tensor.shape[3] // 2])

    # Slice the second half of the last dimension (channels)
    slice2 = tf.slice(tensor, [0, 0, 0, tensor.shape[3] // 2], [-1, -1, -1, tensor.shape[3] // 2])

    return slice1, slice2

def mid_blk(input, filters):
    net = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    net = layers.Conv2D(filters, kernel_size=3, strides=1, padding='valid')(net)
    net = tf.keras.layers.LeakyReLU()(net)
    net = simpl_chan_att(net, filters)
    net2 = input + net
    net = tf.pad(net2, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    net = layers.Conv2D(filters, kernel_size=3, strides=1, padding='valid')(net)
    net = tf.keras.layers.LeakyReLU()(net)
    net = spatial_attention(net)
    return net + net2

def spatial_attention(input_tensor):
    avg_pool = tf.reduce_mean(input_tensor, axis=[3], keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=[3], keepdims=True)
    concat = tf.concat([avg_pool, max_pool], 3)
    attention = layers.Conv2D(1, kernel_size=[7,7], strides=[1,1], padding='same', activation='sigmoid')(concat)
    
    return layers.Multiply()([input_tensor, attention])


def simpl_chan_att(inputs, channels):
    average_pooling = tf.reduce_mean(inputs, axis=[3], keepdims=True)
    features = keras.layers.Conv2D(filters=channels, kernel_size=1)(average_pooling)
    return inputs * features

def DYNUnet(input_shape=(None, None, 3), enc_blocks=[1,1,1,1], dec_blocks=[1,1,1,1], bottom_layers=1, num_filters=32):
    input = layers.Input(input_shape)
    x = input
    skip_connections = []

    for i in list(enc_blocks):
        for _ in range(int(i)):
            x  = encoder_blk(x, num_filters)
        skip_connections.append(x)
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        x = layers.Conv2D(num_filters*2, kernel_size=3, strides=2, padding='valid')(x)

        num_filters = num_filters*2


    for _ in range(int(bottom_layers)):
        x  = mid_blk(x, num_filters)

    j = 1
    for i in list(dec_blocks):
        num_filters = num_filters//2

        tensor_concat = tf.pad(skip_connections[-j], [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        tensor_concat = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='valid')(tensor_concat)
        tensor_concat = tf.keras.layers.LeakyReLU()(tensor_concat)

        net = tf.keras.layers.Conv2DTranspose(num_filters, kernel_size=1, strides=2, padding='same')(x)
        net = tf.keras.layers.LeakyReLU()(net)
        
        x = layers.Add()([net, tensor_concat])

        for _ in range(int(i)):
            x = decoder_blk(x, num_filters)
        j += 1

    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    x = layers.Conv2D(3, kernel_size=3, activation=None, padding='valid')(x)

    model = tf.keras.models.Model(inputs=[input], outputs=[input+x])

    return model