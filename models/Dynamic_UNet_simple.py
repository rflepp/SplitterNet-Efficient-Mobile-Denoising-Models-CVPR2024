"""Lightweight dynamic U-Net baseline."""
import keras
from keras import layers


def conv_lrelu(x, filters):
    x = layers.Conv2D(filters, kernel_size=3, padding="same")(x)
    return layers.LeakyReLU(negative_slope=0.2)(x)


def DYNUnet(input_shape=(None, None, 3), enc_blocks=(2, 2, 2, 2), dec_blocks=(2, 2, 2, 2), bottom_layers=2, num_filters=32):
    inputs = layers.Input(input_shape)
    x = inputs

    skips = []
    for n_blocks in enc_blocks:
        for _ in range(n_blocks):
            x = conv_lrelu(x, num_filters)
        skips.append(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        num_filters *= 2

    for _ in range(bottom_layers):
        x = conv_lrelu(x, num_filters)

    for n_blocks, skip in zip(dec_blocks, reversed(skips)):
        x = layers.Concatenate()([layers.UpSampling2D(size=2)(x), skip])
        num_filters //= 2
        for _ in range(n_blocks):
            x = conv_lrelu(x, num_filters)

    x = layers.Conv2D(3, kernel_size=3, padding="same")(x)
    return keras.Model(inputs=inputs, outputs=x, name="Dynamic_UNet_simple")
