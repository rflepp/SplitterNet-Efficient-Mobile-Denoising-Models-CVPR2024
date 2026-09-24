"""Dynamic PlainNet, following the plain baseline of NAFNet (Chen et al., 2022)."""
import keras
from keras import layers, ops


def double_conv(inputs, filters):
    x = layers.Conv2D(filters, kernel_size=1)(inputs)
    x = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, kernel_size=1)(x)
    middle = inputs + x
    x = layers.Conv2D(filters, kernel_size=1)(middle)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, kernel_size=1)(x)
    return middle + x


def upconv_concat(x, skip, filters):
    x = layers.Conv2D(filters * 2, kernel_size=1)(x)
    return ops.depth_to_space(x, 2) + skip


def DYNUnet(input_shape=(None, None, 3), enc_blocks=(1, 1, 1, 1), dec_blocks=(1, 1, 1, 1), bottom_layers=2, num_filters=32):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(num_filters, kernel_size=3, padding="same")(inputs)

    skips = []
    for n_blocks in enc_blocks:
        for _ in range(n_blocks):
            x = double_conv(x, num_filters)
        skips.append(x)
        num_filters *= 2
        x = layers.Conv2D(num_filters, kernel_size=2, strides=2, padding="same")(x)

    for _ in range(bottom_layers):
        x = double_conv(x, num_filters)

    for n_blocks, skip in zip(dec_blocks, reversed(skips)):
        x = upconv_concat(x, skip, num_filters)
        num_filters //= 2
        for _ in range(n_blocks):
            x = double_conv(x, num_filters)

    x = layers.Conv2D(3, kernel_size=3, padding="same")(x)
    return keras.Model(inputs=inputs, outputs=x, name="Dynamic_PlainNet")
