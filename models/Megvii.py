"""Megvii: winner of the MAI 2021 real-time image denoising challenge."""
import keras
from keras import layers, ops

from .common import split_channels, zero_pad


def encoder_blk(x, filters):
    x = layers.LeakyReLU()(layers.Conv2D(filters // 2, kernel_size=3, padding="same")(x))
    return layers.LeakyReLU()(layers.Conv2D(filters * 2, kernel_size=3, padding="same")(x))


def decoder_blk(x, skip, filters):
    skip = layers.Conv2D(filters, kernel_size=3, padding="same")(skip)
    x = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="valid")(x)
    x = layers.Add()([x, skip])
    x = layers.LeakyReLU()(layers.Conv2D(filters, kernel_size=3, padding="same")(x))

    half_1, half_2 = split_channels(x)
    half_1 = layers.LeakyReLU()(layers.Conv2D(filters // 2, kernel_size=3, padding="same")(half_1))
    half_2 = layers.Conv2D(filters // 2, kernel_size=3, padding="same")(half_2)
    return ops.concatenate([half_1, half_2], axis=3)


def DYNUnet(input_shape=(None, None, 3), enc_blocks=(1, 1, 1, 1), dec_blocks=(1, 1, 1, 1), num_filters=32):
    inputs = layers.Input(input_shape)
    x = layers.LeakyReLU()(layers.Conv2D(8, kernel_size=3, padding="same")(inputs))
    x = layers.LeakyReLU()(layers.Conv2D(num_filters, kernel_size=3, padding="same")(x))

    skips = []
    for n_blocks in enc_blocks:
        skips.append(x)
        x = layers.Conv2D(num_filters, kernel_size=4, strides=2, padding="valid")(zero_pad(x))
        for _ in range(n_blocks):
            x = encoder_blk(x, num_filters)
        num_filters *= 2

    num_filters //= 2
    for n_blocks, skip in zip(dec_blocks, reversed(skips)):
        num_filters //= 2
        for _ in range(n_blocks):
            x = decoder_blk(x, skip, num_filters)

    x = layers.Conv2D(3, kernel_size=3, padding="same")(x)
    return keras.Model(inputs=inputs, outputs=x + inputs, name="Megvii")
