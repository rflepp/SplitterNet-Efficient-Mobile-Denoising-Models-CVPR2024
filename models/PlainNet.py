"""PlainNet: four-level U-Net built from the plain blocks of NAFNet (Chen et al., 2022)."""
import keras
from keras import layers, ops


def double_conv(inputs, filters, pool=True):
    x = layers.Conv2D(filters, kernel_size=1)(inputs)
    x = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters, kernel_size=1)(x)
    middle = inputs + x
    x = layers.Conv2D(filters, kernel_size=1)(middle)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = middle + x

    if pool:
        return x, layers.Conv2D(filters * 2, kernel_size=1, strides=2, padding="same")(x)
    return x


def upconv_concat(x, filters, skip):
    x = layers.Conv2D(filters * 4, kernel_size=1)(x)
    return ops.depth_to_space(x, 2) + skip


def UNet(input_shape=(None, None, 3), base_filters=32):
    s = base_filters
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(s, kernel_size=3, padding="same")(inputs)

    skips = []
    for level in range(4):
        skip, x = double_conv(x, s * 2 ** level)
        skips.append(skip)
    x = double_conv(x, s * 16, pool=False)

    for level in reversed(range(4)):
        x = upconv_concat(x, s * 2 ** level, skips[level])
        x = double_conv(x, s * 2 ** level, pool=False)

    x = layers.Conv2D(3, kernel_size=3, padding="same")(x)
    return keras.Model(inputs=inputs, outputs=x, name="PlainNet")
