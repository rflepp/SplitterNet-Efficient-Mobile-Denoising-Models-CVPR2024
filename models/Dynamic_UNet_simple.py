import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Concatenate, UpSampling2D
import logging

logging.basicConfig(level=logging.DEBUG)

def double_conv(net, filters):
    net = Conv2D(filters, kernel_size=3, padding='same')(net)
    net = tf.nn.leaky_relu(net, alpha=0.2, name=None)
    return net

def upconv_concat(net, tensor_concat):
    net = UpSampling2D(size=(2, 2))(net)
    return Concatenate(axis=-1)([net, tensor_concat])

def DYNUnet(input_shape=(None, None, 3), enc_blocks=[2,2,2,2], dec_blocks=[2,2,2,2], bottom_layers=2, num_filters=32):
    input = Input(input_shape)
    x = input
    skip_connections = []

    for i in list(enc_blocks):
        logging.debug(i)
        for _ in range(int(i)):
            x  = double_conv(x, num_filters)
        skip_connections.append(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        num_filters = num_filters*2

    for _ in range(int(bottom_layers)):
        x  = double_conv(x, num_filters)

    j = 1
    for i in list(dec_blocks):
        logging.debug(i)
        x = upconv_concat(x, skip_connections[-j])
        num_filters = num_filters//2
        for _ in range(int(i)):
            x = double_conv(x, num_filters)
        j += 1

    x = Conv2D(3, kernel_size=3, padding='same', activation=None)(x)
    model = tf.keras.models.Model(inputs=[input], outputs=[x])

    return model
