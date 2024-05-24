import tensorflow as tf
from tensorflow.keras import layers
import logging

logging.basicConfig(level=logging.DEBUG)

def Unet(input_shape=(None, None, 3), num_filters=16):
    input = layers.Input(input_shape)

    x1 = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='relu')(input)
    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x1)
    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    add1 = layers.Add()([x1, x])

    x2 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='same')(add1)
    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x2)
    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    add2 = layers.Add()([x2, x])

    x3 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='same')(add2)
    x = layers.Conv2D(num_filters*2, kernel_size=3, strides=1, padding='same', activation='relu')(x3)
    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    add3 = layers.Add()([x3, x])

    x = tf.keras.layers.Conv2DTranspose(filters=num_filters, kernel_size=1, strides=2, padding='same')(add3)

    concat1 = tf.concat([add2, x], axis=3)

    x4 = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='relu')(concat1)
    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x4)
    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    add4 = layers.Add()([x4, x])

    x = tf.keras.layers.Conv2DTranspose(filters=num_filters, kernel_size=1, strides=2, padding='same')(add4)

    concat2 = tf.concat([add1, x], axis=3)

    x5 = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='relu')(concat2)
    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x5)
    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    add5 = layers.Add()([x5, x])

    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='relu')(add5)

    x = layers.Conv2D(3, kernel_size=3, padding='same', activation=None)(x)
    model = tf.keras.models.Model(inputs=[input], outputs=[input+x])

    return model