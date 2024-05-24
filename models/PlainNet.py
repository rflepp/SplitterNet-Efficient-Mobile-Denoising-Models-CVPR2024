import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model

def double_conv(input, filters, pool=True):
    net = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(input)
    net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(net) # native to be used?
    net = layers.LeakyReLU()(net)
    net = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(net)
    middle = input + net
    net = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(middle)
    net = layers.LeakyReLU()(net)
    net = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(net)
    net = middle + net

    if pool:
        return net, layers.Conv2D(filters*2, kernel_size=1, strides=2, padding='same')(net)
    else:
        return net


def upconv_concat(net, filters, res_con):
    net = layers.Conv2D(filters*4, kernel_size=1, padding='VALID', activation=None, strides=1)(net)
    net = tf.nn.depth_to_space(net, block_size=2, data_format='NHWC')
    net = net + res_con
    return net


def UNet(input_size, base_filters=32):

    s = base_filters
    image = keras.Input(shape=input_size)
    conv0 = layers.Conv2D(filters=s, kernel_size=3, padding="same")(image)

    conv1, pool1 = double_conv(conv0, s)
    conv2, pool2 = double_conv(pool1, s*2)
    conv3, pool3 = double_conv(pool2, s*4)
    conv4, pool4 = double_conv(pool3, s*8)
    conv5 = double_conv(pool4, s*16, pool=False)

    up6 = upconv_concat(conv5, s*8, conv4)
    conv6 = double_conv(up6, s*8, pool=False)

    up7 = upconv_concat(conv6, s*4, conv3)
    conv7 = double_conv(up7, s*4, pool=False)

    up8 = upconv_concat(conv7, s*2, conv2)
    conv8 = double_conv(up8, s*2, pool=False)

    up9 = upconv_concat(conv8, s, conv1)
    conv9 = double_conv(up9, s, pool=False)

    conv_last = layers.Conv2D(3, kernel_size=3, padding='same', activation=None)(conv9)

    model = Model(inputs=[image], outputs=[conv_last])

    return model
    