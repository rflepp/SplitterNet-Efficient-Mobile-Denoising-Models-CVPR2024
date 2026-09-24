"""NOAHTCV: runner-up of the MAI 2021 real-time image denoising challenge."""
import keras
from keras import layers


def conv(x, filters, strides=1, activation="relu"):
    return layers.Conv2D(filters, kernel_size=3, strides=strides, padding="same", activation=activation)(x)


def res_blk(x, mid_filters, filters):
    y = conv(x, mid_filters)
    y = conv(y, filters)
    return layers.Add()([x, y])


def Unet(input_shape=(None, None, 3), num_filters=16):
    inputs = layers.Input(input_shape)

    x1 = conv(inputs, num_filters)
    y = conv(x1, num_filters)
    y = conv(y, num_filters)
    add1 = layers.Add()([x1, y])

    add2 = res_blk(conv(add1, num_filters, strides=2, activation=None), num_filters, num_filters)
    add3 = res_blk(conv(add2, num_filters, strides=2, activation=None), num_filters * 2, num_filters)

    x = layers.Conv2DTranspose(num_filters, kernel_size=1, strides=2, padding="same")(add3)
    x4 = conv(layers.Concatenate()([add2, x]), num_filters)
    add4 = res_blk(x4, num_filters, num_filters)

    x = layers.Conv2DTranspose(num_filters, kernel_size=1, strides=2, padding="same")(add4)
    x5 = conv(layers.Concatenate()([add1, x]), num_filters)
    add5 = res_blk(x5, num_filters, num_filters)

    x = conv(add5, num_filters)
    x = conv(x, 3, activation=None)
    return keras.Model(inputs=inputs, outputs=inputs + x, name="NOAHTCV")
