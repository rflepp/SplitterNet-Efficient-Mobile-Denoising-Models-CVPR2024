import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow import keras

def encoder_blk(input):
    split_tensors = split_using_slice(input)
    split_tensor_1 = split_tensors[0]
    split_tensor_2 = split_tensors[1]
    return split_tensor_1, split_tensor_2

def split_using_slice(tensor):
    slice1 = tf.slice(tensor, [0, 0, 0, 0], [-1, -1, -1, tensor.shape[3] // 2])
    slice2 = tf.slice(tensor, [0, 0, 0, tensor.shape[3] // 2], [-1, -1, -1, tensor.shape[3] // 2])

    return slice1, slice2

def decoder_blk(conv_tensor_1, conv_tensor_2, split_tensor, filters):
    concatenated_tensor = tf.concat([conv_tensor_1, conv_tensor_2], axis=3)
    concatenated_tensor = layers.Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same')(concatenated_tensor)
    concatenated_tensor = layers.LeakyReLU()(concatenated_tensor)
    return concatenated_tensor+split_tensor

def mid_blk(input, filters):
    net = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    net = layers.Conv2D(filters, kernel_size=3, strides=1, padding='valid')(net)
    net = layers.LeakyReLU()(net)
    net = simpl_chan_att(net, filters)
    net2 = net + input
    net = tf.pad(net2, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    net = layers.Conv2D(filters, kernel_size=3, strides=1, padding='valid')(net)
    net = layers.LeakyReLU()(net)
    net = spatial_attention(net)
    return net+net2

def spatial_attention(input_tensor):
    avg_pool = tf.reduce_mean(input_tensor, axis=[3], keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=[3], keepdims=True)
    concat = tf.concat([avg_pool, max_pool], 3)
    net = tf.pad(concat, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    attention = layers.Conv2D(1, kernel_size=[3,3], strides=[1,1], padding='valid', activation='sigmoid')(net)
    
    return layers.Multiply()([input_tensor, attention])

def simpl_chan_att(inputs, channels):
    average_pooling = keras.layers.GlobalAveragePooling2D()(inputs)
    feature_descriptor = tf.reshape(
        average_pooling, shape=(-1, 1, 1, channels)
    )
    features = keras.layers.Conv2D(filters=channels, kernel_size=1)(feature_descriptor)
    return inputs * features


def DYNUnet(input_shape=(None, None, 3), num_filters=32):
    input = layers.Input(input_shape)
    x = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same')(input)

    # Encoder
    # 1st stage 
    # 256x256x16
    split_tensor_1, split_tensor_2  = encoder_blk(x)
    # 128x128x32
    split_tensor_1 = tf.pad(split_tensor_1, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_1 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_1)
    conv_tensor_1 = layers.LeakyReLU()(conv_tensor_1)
    split_tensor_2 = tf.pad(split_tensor_2, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_2 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_2)
    conv_tensor_2 = layers.LeakyReLU()(conv_tensor_2)

    # 2nd stage 
    # 128x128x16
    split_tensor_3, split_tensor_4  = encoder_blk(conv_tensor_1)
    split_tensor_5, split_tensor_6  = encoder_blk(conv_tensor_2)
    # 64x64x32
    split_tensor_3 = tf.pad(split_tensor_3, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_3 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_3)
    conv_tensor_3 = layers.LeakyReLU()(conv_tensor_3)
    split_tensor_4 = tf.pad(split_tensor_4, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_4 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_4)
    conv_tensor_4 = layers.LeakyReLU()(conv_tensor_4)
    split_tensor_5 = tf.pad(split_tensor_5, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_5 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_5)
    conv_tensor_5 = layers.LeakyReLU()(conv_tensor_5)
    split_tensor_6 = tf.pad(split_tensor_6, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_6 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_6)
    conv_tensor_6 = layers.LeakyReLU()(conv_tensor_6)


    # 3rd stage 
    # 64x64x16
    split_tensor_8, split_tensor_9  = encoder_blk(conv_tensor_3)
    split_tensor_10, split_tensor_11  = encoder_blk(conv_tensor_4)
    split_tensor_12, split_tensor_13  = encoder_blk(conv_tensor_5)
    split_tensor_14, split_tensor_15  = encoder_blk(conv_tensor_6)
    # 32x32x32
    split_tensor_8 = tf.pad(split_tensor_8, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_8 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_8)
    conv_tensor_8 = layers.LeakyReLU()(conv_tensor_8)
    split_tensor_9 = tf.pad(split_tensor_9, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_9 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_9)
    conv_tensor_9 = layers.LeakyReLU()(conv_tensor_9)
    split_tensor_10 = tf.pad(split_tensor_10, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_10 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_10)
    conv_tensor_10 = layers.LeakyReLU()(conv_tensor_10)
    split_tensor_11 = tf.pad(split_tensor_11, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_11 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_11)
    conv_tensor_11 = layers.LeakyReLU()(conv_tensor_11)
    split_tensor_12 = tf.pad(split_tensor_12, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_12 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_12)
    conv_tensor_12 = layers.LeakyReLU()(conv_tensor_12)
    split_tensor_13 = tf.pad(split_tensor_13, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_13 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_13)
    conv_tensor_13 = layers.LeakyReLU()(conv_tensor_13)
    split_tensor_14 = tf.pad(split_tensor_14, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_14 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_14)
    conv_tensor_14 = layers.LeakyReLU()(conv_tensor_14)
    split_tensor_15 = tf.pad(split_tensor_15, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_15 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_15)
    conv_tensor_15 = layers.LeakyReLU()(conv_tensor_15)

    # 4th stage
    # 32x32x16
    split_tensor_16, split_tensor_17  = encoder_blk(conv_tensor_8)
    split_tensor_18, split_tensor_19  = encoder_blk(conv_tensor_9)
    split_tensor_20, split_tensor_21  = encoder_blk(conv_tensor_10)
    split_tensor_22, split_tensor_23  = encoder_blk(conv_tensor_11)
    split_tensor_24, split_tensor_25  = encoder_blk(conv_tensor_12)
    split_tensor_26, split_tensor_27  = encoder_blk(conv_tensor_13)
    split_tensor_28, split_tensor_29  = encoder_blk(conv_tensor_14)
    split_tensor_30, split_tensor_31  = encoder_blk(conv_tensor_15)
    # 16x16x32
    split_tensor_16 = tf.pad(split_tensor_16, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_16 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_16)
    conv_tensor_16 = layers.LeakyReLU()(conv_tensor_16)
    split_tensor_17 = tf.pad(split_tensor_17, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_17 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_17)
    conv_tensor_17 = layers.LeakyReLU()(conv_tensor_17)
    split_tensor_18 = tf.pad(split_tensor_18, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_18 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_18)
    conv_tensor_18 = layers.LeakyReLU()(conv_tensor_18)
    split_tensor_19 = tf.pad(split_tensor_19, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_19 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_19)
    conv_tensor_19 = layers.LeakyReLU()(conv_tensor_19)
    split_tensor_20 = tf.pad(split_tensor_20, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_20 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_20)
    conv_tensor_20 = layers.LeakyReLU()(conv_tensor_20)
    split_tensor_21 = tf.pad(split_tensor_21, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_21 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_21)
    conv_tensor_21 = layers.LeakyReLU()(conv_tensor_21)
    split_tensor_22 = tf.pad(split_tensor_22, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_22 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_22)
    conv_tensor_22 = layers.LeakyReLU()(conv_tensor_22)
    split_tensor_23 = tf.pad(split_tensor_23, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_23 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_23)
    conv_tensor_23 = layers.LeakyReLU()(conv_tensor_23)
    split_tensor_24 = tf.pad(split_tensor_24, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_24 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_24)
    conv_tensor_24 = layers.LeakyReLU()(conv_tensor_24)
    split_tensor_25 = tf.pad(split_tensor_25, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_25 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_25)
    conv_tensor_25 = layers.LeakyReLU()(conv_tensor_25)
    split_tensor_26 = tf.pad(split_tensor_26, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_26 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_26)
    conv_tensor_26 = layers.LeakyReLU()(conv_tensor_26)
    split_tensor_27 = tf.pad(split_tensor_27, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_27 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_27)
    conv_tensor_27 = layers.LeakyReLU()(conv_tensor_27)
    split_tensor_28 = tf.pad(split_tensor_28, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_28 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_28)
    conv_tensor_28 = layers.LeakyReLU()(conv_tensor_28)
    split_tensor_29 = tf.pad(split_tensor_29, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_29 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_29)
    conv_tensor_29 = layers.LeakyReLU()(conv_tensor_29)
    split_tensor_30 = tf.pad(split_tensor_30, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_30 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_30)
    conv_tensor_30 = layers.LeakyReLU()(conv_tensor_30)
    split_tensor_31 = tf.pad(split_tensor_31, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    conv_tensor_31 = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='valid')(split_tensor_31)
    conv_tensor_31 = layers.LeakyReLU()(conv_tensor_31)


    # MidBlock
    # 32x32x32
    conv_tensor_16 = mid_blk(conv_tensor_16, num_filters)
    conv_tensor_17 = mid_blk(conv_tensor_17, num_filters)
    conv_tensor_18 = mid_blk(conv_tensor_18, num_filters)
    conv_tensor_19 = mid_blk(conv_tensor_19, num_filters)
    conv_tensor_20 = mid_blk(conv_tensor_20, num_filters)
    conv_tensor_21 = mid_blk(conv_tensor_21, num_filters)
    conv_tensor_22 = mid_blk(conv_tensor_22, num_filters)
    conv_tensor_23 = mid_blk(conv_tensor_23, num_filters)
    conv_tensor_24 = mid_blk(conv_tensor_24, num_filters)
    conv_tensor_25 = mid_blk(conv_tensor_25, num_filters)
    conv_tensor_26 = mid_blk(conv_tensor_26, num_filters)
    conv_tensor_27 = mid_blk(conv_tensor_27, num_filters)
    conv_tensor_28 = mid_blk(conv_tensor_28, num_filters)
    conv_tensor_29 = mid_blk(conv_tensor_29, num_filters)
    conv_tensor_30 = mid_blk(conv_tensor_30, num_filters)
    conv_tensor_31 = mid_blk(conv_tensor_31, num_filters)

    # Decoder
    # 4th stage
    fusion_tensor_1 = decoder_blk(conv_tensor_16, conv_tensor_17, conv_tensor_8, num_filters)
    fusion_tensor_2 = decoder_blk(conv_tensor_18, conv_tensor_19, conv_tensor_9, num_filters)
    fusion_tensor_3 = decoder_blk(conv_tensor_20, conv_tensor_21, conv_tensor_10, num_filters)
    fusion_tensor_4 = decoder_blk(conv_tensor_22, conv_tensor_23, conv_tensor_11, num_filters)
    fusion_tensor_5 = decoder_blk(conv_tensor_24, conv_tensor_25, conv_tensor_12, num_filters)
    fusion_tensor_6 = decoder_blk(conv_tensor_26, conv_tensor_27, conv_tensor_13, num_filters)
    fusion_tensor_7 = decoder_blk(conv_tensor_28, conv_tensor_29, conv_tensor_14, num_filters)
    fusion_tensor_8 = decoder_blk(conv_tensor_30, conv_tensor_31, conv_tensor_15, num_filters)

    # 3rd stage
    # 64x64x32
    fusion_tensor_9 = decoder_blk(fusion_tensor_1, fusion_tensor_2, conv_tensor_3, num_filters)
    fusion_tensor_10 = decoder_blk(fusion_tensor_3, fusion_tensor_4, conv_tensor_4, num_filters)
    fusion_tensor_11 = decoder_blk(fusion_tensor_5, fusion_tensor_6, conv_tensor_5, num_filters)
    fusion_tensor_12 = decoder_blk(fusion_tensor_7, fusion_tensor_8, conv_tensor_6, num_filters)

    # 2nd stage
    # 128x128x32
    fusion_tensor_13 = decoder_blk(fusion_tensor_9, fusion_tensor_10, conv_tensor_1, num_filters)
    fusion_tensor_14 = decoder_blk(fusion_tensor_11, fusion_tensor_12, conv_tensor_2, num_filters)

    # 1st stage:
    fusion_tensor_15 = decoder_blk(fusion_tensor_13, fusion_tensor_14, x, num_filters)

    fusion_tensor_15 = layers.Conv2D(3, kernel_size=3, activation=None, padding='same')(fusion_tensor_15)
    model = Model(inputs=[input], outputs=[input+fusion_tensor_15])

    return model