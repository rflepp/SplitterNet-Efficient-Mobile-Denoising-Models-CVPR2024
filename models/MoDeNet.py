"""MoDeNet: dynamic multi-scale efficient denoiser (Flepp et al., CVPR 2024)."""
import keras
from keras import layers, ops

from .common import reflect_pad, split_channels


def conv_lrelu(x, filters, strides=1):
    x = layers.Conv2D(filters, kernel_size=3, strides=strides, padding="valid")(reflect_pad(x))
    return layers.LeakyReLU()(x)


def decoder_blk(x, filters):
    x = conv_lrelu(x, filters)
    half_1, half_2 = split_channels(x)
    return ops.concatenate([conv_lrelu(half_1, filters), conv_lrelu(half_2, filters)], axis=3)


def mid_blk(inputs, filters):
    x = conv_lrelu(inputs, filters)
    residual = inputs + simpl_chan_att(x, filters)
    x = conv_lrelu(residual, filters)
    return spatial_attention(x) + residual


def spatial_attention(inputs):
    avg_pool = ops.mean(inputs, axis=3, keepdims=True)
    max_pool = ops.max(inputs, axis=3, keepdims=True)
    x = ops.concatenate([avg_pool, max_pool], axis=3)
    attention = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(x)
    return layers.Multiply()([inputs, attention])


def simpl_chan_att(inputs, channels):
    x = ops.mean(inputs, axis=3, keepdims=True)
    return inputs * layers.Conv2D(channels, kernel_size=1)(x)


def DYNUnet(input_shape=(None, None, 3), enc_blocks=(1, 1, 1, 1), dec_blocks=(1, 1, 1, 1), bottom_layers=1, num_filters=32):
    inputs = layers.Input(input_shape)
    x = inputs

    skips = []
    for n_blocks in enc_blocks:
        for _ in range(n_blocks):
            x = conv_lrelu(x, num_filters)
        skips.append(x)
        num_filters *= 2
        x = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding="valid")(reflect_pad(x))

    for _ in range(bottom_layers):
        x = mid_blk(x, num_filters)

    for n_blocks, skip in zip(dec_blocks, reversed(skips)):
        num_filters //= 2
        skip = conv_lrelu(skip, num_filters)
        x = layers.Conv2DTranspose(num_filters, kernel_size=1, strides=2, padding="same")(x)
        x = layers.Add()([layers.LeakyReLU()(x), skip])
        for _ in range(n_blocks):
            x = decoder_blk(x, num_filters)

    x = layers.Conv2D(3, kernel_size=3, padding="valid")(reflect_pad(x))
    return keras.Model(inputs=inputs, outputs=inputs + x, name="MoDeNet")
