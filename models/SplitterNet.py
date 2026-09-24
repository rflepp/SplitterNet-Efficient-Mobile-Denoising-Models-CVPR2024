"""SplitterNet: efficient mobile image denoising (Flepp et al., CVPR 2024)."""
import keras
from keras import layers, ops

from .common import reflect_pad, split_channels

NUM_LEVELS = 4


@keras.saving.register_keras_serializable(package="SplitterNet")
class ChannelLayerNorm(layers.Layer):
    """Layer normalisation over the channel axis, written with elementary ops for TFLite."""

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name="gamma", shape=input_shape[-1:], initializer="ones")
        self.beta = self.add_weight(name="beta", shape=input_shape[-1:], initializer="zeros")

    def call(self, x):
        mean, variance = ops.moments(x, axes=[-1], keepdims=True)
        return self.gamma * (x - mean) / ops.sqrt(variance + self.epsilon) + self.beta

    def get_config(self):
        return {**super().get_config(), "epsilon": self.epsilon}


def down_blk(x, filters):
    x = layers.Conv2D(filters, kernel_size=3, strides=2, padding="valid")(reflect_pad(x))
    return layers.LeakyReLU()(x)


def decoder_blk(branch_1, branch_2, skip, filters, layer_norm=False):
    x = ops.concatenate([branch_1, branch_2], axis=3)
    if layer_norm:
        x = ChannelLayerNorm()(x)
    x = layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding="same")(x)
    return layers.LeakyReLU()(x) + skip


def mid_blk(inputs, filters, layer_norm=False):
    x = ChannelLayerNorm()(inputs) if layer_norm else inputs
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding="valid")(reflect_pad(x))
    x = layers.LeakyReLU()(x)
    residual = simpl_chan_att(x, filters) + inputs
    x = ChannelLayerNorm()(residual) if layer_norm else residual
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding="valid")(reflect_pad(x))
    x = layers.LeakyReLU()(x)
    return spatial_attention(x) + residual


def spatial_attention(inputs):
    avg_pool = ops.mean(inputs, axis=3, keepdims=True)
    max_pool = ops.max(inputs, axis=3, keepdims=True)
    x = reflect_pad(ops.concatenate([avg_pool, max_pool], axis=3))
    attention = layers.Conv2D(1, kernel_size=3, strides=1, padding="valid", activation="sigmoid")(x)
    return layers.Multiply()([inputs, attention])


def simpl_chan_att(inputs, channels):
    x = layers.GlobalAveragePooling2D(keepdims=True)(inputs)
    return inputs * layers.Conv2D(channels, kernel_size=1)(x)


def DYNUnet(input_shape=(None, None, 3), num_filters=32, layer_norm=False):
    """Build SplitterNet.

    Args:
        input_shape: Input shape (H, W, C). H and W must be divisible by 16.
        num_filters: Number of filters used by every convolution.
        layer_norm: Add layer normalisation before each split, mid block and decoder block.
    """
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding="same")(inputs)

    # Encoder: at every level each branch is split along the channels and both
    # halves are downsampled independently, doubling the number of branches.
    skips = []
    branches = [x]
    for _ in range(NUM_LEVELS):
        skips.append(branches)
        if layer_norm:
            branches = [layers.LayerNormalization()(b) for b in branches]
        halves = [half for b in branches for half in split_channels(b)]
        branches = [down_blk(half, num_filters) for half in halves]

    branches = [mid_blk(b, num_filters, layer_norm) for b in branches]

    # Decoder: fuse neighbouring branch pairs and add the matching encoder skip.
    for level_skips in reversed(skips):
        branches = [decoder_blk(branches[2 * i], branches[2 * i + 1], skip, num_filters, layer_norm)
                    for i, skip in enumerate(level_skips)]

    out = layers.Conv2D(3, kernel_size=3, padding="same")(branches[0])
    return keras.Model(inputs=inputs, outputs=inputs + out, name="SplitterNet_LN" if layer_norm else "SplitterNet")
