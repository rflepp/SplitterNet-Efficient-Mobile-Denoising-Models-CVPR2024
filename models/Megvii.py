import tensorflow as tf
from tensorflow.keras import layers
import math
import logging

logging.basicConfig(level=logging.DEBUG)

def encoder_blk(input, filters):
    net = layers.Conv2D(filters//2, kernel_size=3, strides=1, padding='same')(input)
    net = layers.LeakyReLU()(net)
    net = layers.Conv2D(filters*2, kernel_size=3, strides=1, padding='same')(net)
    net = layers.LeakyReLU()(net)
    return net

def decoder_blk(input, tensor_concat, filters):
    tensor_concat = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(tensor_concat)

    net = layers.Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='valid', use_bias=True)(input)

    net = layers.Add()([net, tensor_concat])

    net = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(net)
    net = layers.LeakyReLU()(net)

    split_tensor_1, split_tensor_2 = split_using_slice(net)

    split_tensor_1 = layers.Conv2D(filters//2, kernel_size=3, strides=1, padding='same')(split_tensor_1)
    split_tensor_1 = layers.LeakyReLU()(split_tensor_1)
    split_tensor_2 = layers.Conv2D(filters//2, kernel_size=3, strides=1, padding='same')(split_tensor_2)


    concatenated_tensor = tf.concat([split_tensor_1, split_tensor_2], axis=3)

    return concatenated_tensor

def split_using_slice(tensor):
    # Slice the first half of the last dimension (channels)
    slice1 = tf.slice(tensor, [0, 0, 0, 0], [-1, -1, -1, tensor.shape[3] // 2])

    # Slice the second half of the last dimension (channels)
    slice2 = tf.slice(tensor, [0, 0, 0, tensor.shape[3] // 2], [-1, -1, -1, tensor.shape[3] // 2])

    return slice1, slice2

def DYNUnet(input_shape=(None, None, 3), enc_blocks=[1,1,1,1], dec_blocks=[1,1,1,1], num_filters=32):
    input = layers.Input(input_shape)
    x = input
    skip_connections = []

    x = layers.Conv2D(8, kernel_size=3, strides=1, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same')(x)
    x1 = layers.LeakyReLU()(x)
    x = x1

    for i in list(enc_blocks):
        skip_connections.append(x)
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
        x = layers.Conv2D(num_filters, kernel_size=4, strides=2, padding='valid')(x)
        for _ in range(int(i)):
            x  = encoder_blk(x, num_filters)

        num_filters = num_filters*2

    num_filters = num_filters//2
    j = 1
    for i in list(dec_blocks):
        num_filters = num_filters//2
        for _ in range(int(i)):
            x = decoder_blk(x, skip_connections[-j], num_filters)
        j += 1

    x = layers.Conv2D(3, kernel_size=3, padding='same', activation=None)(x)

    model = tf.keras.models.Model(inputs=[input], outputs=[x+input])

    return model