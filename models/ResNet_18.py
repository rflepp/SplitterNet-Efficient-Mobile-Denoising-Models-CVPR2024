"""ResNet-18 encoder with a U-Net style decoder, used as a denoising baseline."""
import keras
from keras import layers


def conv_block(x, filters, kernel_size=3, stride=1):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def identity_block(inputs, filters):
    x = conv_block(inputs, filters)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(layers.Add()([x, inputs]))


def resnet_block(x, filters, num_blocks, downsample=True):
    x = conv_block(x, filters, stride=2 if downsample else 1)
    for _ in range(num_blocks - 1):
        x = identity_block(x, filters)
    return x


def upsample_concat_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    return layers.Conv2DTranspose(filters, 3, strides=1, padding="same")(x)


def ResNet18_Denoiser(input_shape=(None, None, 3)):
    inputs = layers.Input(input_shape)

    # Encoder
    x = conv_block(inputs, 64, stride=2)
    skip1 = identity_block(x, 64)
    x = resnet_block(skip1, 128, 2)
    skip2 = identity_block(x, 128)
    x = resnet_block(skip2, 256, 2)
    skip3 = identity_block(x, 256)
    x = resnet_block(skip3, 512, 2)
    x = identity_block(x, 512)

    # Decoder
    for skip, filters in ((skip3, 256), (skip2, 128), (skip1, 64), (inputs, 64)):
        x = upsample_concat_block(x, skip, filters)
        x = identity_block(x, filters)

    outputs = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="ResNet_18")
