from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Reshape, Conv1D, MaxPooling1D, \
                                    GlobalAveragePooling1D, Multiply, Add, Lambda, Concatenate, GlobalMaxPooling1D, Flatten, \
                                    UpSampling1D, Conv1DTranspose, BatchNormalization, AveragePooling1D
import math


def channel_attention(input_feature, ratio):
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
    else:
        channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel // ratio, activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal',
                             use_bias=True, bias_initializer='zeros')

    avg_pool = GlobalAveragePooling1D()(input_feature)
    avg_pool = Reshape((1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling1D()(input_feature)
    max_pool = Reshape((1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return Multiply()([input_feature, cbam_feature])


def spatial_attention(input_feature):
    k_size = 7

    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_feature)
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_feature)
    concat = Concatenate(axis=-1)([max_pool, avg_pool])
    spatial_weights = Conv1D(filters=1, kernel_size=k_size, strides=1,
                             padding="same", use_bias=False)(concat)
    spatial_weights = Activation('sigmoid')(spatial_weights)

    return Multiply()([input_feature, spatial_weights])


def CBAM_block(input_feature):
    weighted_map = channel_attention(input_feature, ratio=8)
    weighted_map = spatial_attention(weighted_map)

    return weighted_map


def ResidualCBAMBlock(input_feature, out_dim):
    residual = input_feature

    x = Conv1D(out_dim, 3, padding='same')(input_feature)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(out_dim, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = CBAM_block(x)

    if x.shape[-1] != input_feature.shape[-1]:
        residual = Conv1D(out_dim, 1, padding='same')(residual)
        residual = BatchNormalization()(residual)
    x = Add()([x, residual])
    x = Activation('relu')(x)

    return x


def SpinDOSFeaturizer(input_feature, out_dims):
    x = Conv1D(out_dims[0], kernel_size=20, strides=2, padding='same')(input_feature)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling1D(pool_size=2, strides=2, padding='same')(x)

    # Conv Block1
    x = ResidualCBAMBlock(x, out_dims[0])
    x = ResidualCBAMBlock(x, out_dims[0])
    x = AveragePooling1D(pool_size=2, strides=2, padding='same')(x)

    # Conv Block2
    x = ResidualCBAMBlock(x, out_dims[1])
    x = ResidualCBAMBlock(x, out_dims[1])
    x = AveragePooling1D(pool_size=2, strides=2, padding='same')(x)

    # Conv Block3
    x = ResidualCBAMBlock(x, out_dims[2])
    x = ResidualCBAMBlock(x, out_dims[2])
    x = AveragePooling1D(pool_size=2, strides=2, padding='same')(x)

    # Conv Block4
    x = ResidualCBAMBlock(x, out_dims[3])
    x = ResidualCBAMBlock(x, out_dims[3])

    return x