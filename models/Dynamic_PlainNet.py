import tensorflow as tf
from tensorflow.keras import layers
import logging

logging.basicConfig(level=logging.DEBUG)

def double_conv(input, filters):
    net = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(input)
    net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(net)
    net = layers.Activation('relu')(net)
    net = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(net)
    middle = input + net
    net = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(middle)
    net = layers.Activation('relu')(net)
    net = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(net)
    net = middle + net
    return net

def upconv_concat(net, tensor_concat, filters):
    net = layers.Conv2D(filters*2, kernel_size=1)(net)
    net = PixelShuffle(2)(net)
    return net + tensor_concat

def DYNUnet(input_shape=(None, None, 3), enc_blocks=[1,1,1,1], dec_blocks=[1,1,1,1], bottom_layers=2, num_filters=32):
    input = layers.Input(input_shape)

    x = layers.Conv2D(num_filters, kernel_size=3, padding='SAME', activation=None,strides=1)(input)
    skip_connections = []

    for i in list(enc_blocks):
        logging.debug(i)
        for _ in range(int(i)):
            x  = double_conv(x, num_filters)
        skip_connections.append(x)
        x = layers.Conv2D(num_filters*2, kernel_size=2, strides=2, padding='same')(x)
        num_filters = num_filters*2

    for _ in range(int(bottom_layers)):
        logging.debug("middle")
        x  = double_conv(x, num_filters)

    j = 1
    for i in list(dec_blocks):
        logging.debug(i)
        logging.debug(num_filters)
        x = upconv_concat(x, skip_connections[-j], num_filters)
        num_filters = num_filters//2
        for _ in range(int(i)):
            x = double_conv(x, num_filters)
        j += 1

    x = layers.Conv2D(3, kernel_size=3, padding='same', activation=None)(x)
    model = tf.keras.models.Model(inputs=[input], outputs=[x])

    return model

class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, upscale_factor: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.upscale_factor = upscale_factor

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return tf.nn.depth_to_space(inputs, self.upscale_factor)

    def get_config(self) -> dict:
        """Add upscale factor to the config"""
        config = super().get_config()
        config.update({"upscale_factor": self.upscale_factor})
        return config
