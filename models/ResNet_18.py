import tensorflow as tf
from tensorflow.keras import layers, models, Input

def conv_block(input_tensor, num_filters, kernel_size=3, stride=1):
    x = layers.Conv2D(num_filters, kernel_size, strides=stride, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def identity_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, input_tensor])
    x = layers.ReLU()(x)
    return x

def resnet_block(input_tensor, num_filters, num_blocks, downsample=True):
    x = conv_block(input_tensor, num_filters, stride=2 if downsample else 1)
    for _ in range(num_blocks - 1):
        x = identity_block(x, num_filters)
    return x

def upsample_concat_block(input_tensor, skip_tensor, num_filters):
    x = layers.Conv2DTranspose(num_filters, (3, 3), strides=(2, 2), padding='same')(input_tensor)
    x = layers.concatenate([x, skip_tensor])
    x = layers.Conv2DTranspose(num_filters, (3, 3), strides=(1, 1), padding='same')(x)
    return x

def ResNet18_Denoiser(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder (ResNet-18)
    x = conv_block(inputs, 64, stride=2)
    skip1 = identity_block(x, 64)
    x = resnet_block(skip1, 128, 2)
    skip2 = identity_block(x, 128)
    x = resnet_block(skip2, 256, 2)
    skip3 = identity_block(x, 256)
    x = resnet_block(skip3, 512, 2)
    encoded = identity_block(x, 512)

    # Decoder
    x = upsample_concat_block(encoded, skip3, 256)
    x = identity_block(x, 256)
    x = upsample_concat_block(x, skip2, 128)
    x = identity_block(x, 128)
    x = upsample_concat_block(x, skip1, 64)
    x = identity_block(x, 64)

    # No downsampling in the last block
    x = upsample_concat_block(x, inputs, 64)
    x = identity_block(x, 64)

    # Output layer
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Create model
    model = models.Model(inputs, outputs)
    return model

