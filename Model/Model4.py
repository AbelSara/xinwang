from keras.layers import *
from keras.models import Model
from keras import regularizers
from Model.Sub_block import CenterLossLayer, Attention
from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHeadAttention
from attention import attention_3d_block
import keras.backend as K


def squeeze_excitation_layer(x, out_dim, ratio):
    '''
    SE module performs inter-channel weighting.
    '''
    squeeze = GlobalAveragePooling2D()(x)

    excitation = Dense(out_dim //ratio)(squeeze)
    excitation = ReLU()(excitation)
    excitation = Dense(out_dim, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, out_dim))(excitation)

    scale = multiply([x, excitation])

    return scale


def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             # kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             #kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          # kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def Net():
    input1 = Input(shape=(60, 8, 1))
    input2 = Input(shape=(19, ))
    #X = GaussianNoise(stddev=0.1, training=True)(input1)
    X = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               # kernel_initializer='he_uniform',
               # bias_initializer='zeros'
               # kernel_regularizer=regularizers.l2(1e-5)
               )(input1)
    X = LeakyReLU(0.01)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    X = squeeze_excitation_layer(X, out_dim=64, ratio=8)
    #X = cbam_block(X)

    X = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               # kernel_initializer='he_uniform',
               # bias_initializer='zeros'
               # kernel_regularizer=regularizers.l2(1e-5)
               )(X)
    X = LeakyReLU(0.01)(X)
    X = squeeze_excitation_layer(X, out_dim=64, ratio=4)
    X = MaxPooling2D((2, 2))(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    #X = cbam_block(X)

    X = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding='same',
               # kernel_initializer='he_uniform',
               # bias_initializer='zeros'
               # kernel_regularizer=regularizers.l2(1e-5)
               )(X)
    X = LeakyReLU(0.01)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    X = squeeze_excitation_layer(X, out_dim=128, ratio=8)
    #X = cbam_block(X)

    X = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding='same',
               # kernel_initializer='he_uniform',
               # bias_initializer='zeros'
               # kernel_regularizer=regularizers.l2(1e-5)
               )(X)
    X = LeakyReLU(0.01)(X)
    X = squeeze_excitation_layer(X, out_dim=128, ratio=8)
    X = MaxPooling2D((1, 2))(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    #X = cbam_block(X)

    X = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding='same',
               # kernel_initializer='he_uniform',
               # bias_initializer='zeros'
               # kernel_regularizer=regularizers.l2(1e-5)
               )(X)
    X = LeakyReLU(0.01)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    X = squeeze_excitation_layer(X, out_dim=256, ratio=8)
    #X = cbam_block(X)

    X = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding='same',
               # kernel_initializer='he_uniform',
               # bias_initializer='zeros'
               # kernel_regularizer=regularizers.l2(1e-5)
               )(X)
    X = LeakyReLU(0.01)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    X = squeeze_excitation_layer(X, out_dim=512, ratio=8)
    #X = cbam_block(X)
    X = Reshape((-1, 2*512))(X)

    # X = attention_3d_block(X)
    # X = SeqSelfAttention(attention_activation='sigmoid')(X)
    # X = Dropout(0.3)(X)
    # X = Dense(128, activation='relu')(X)
    #X = GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(X)
    #X = GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(X)

    X = GlobalAveragePooling1D()(X)

    # X = Dense(128)(X)
    # X = LeakyReLU(0.01)(X)
    # X = BatchNormalization()(X)
    # X = Attention(30)(X)
    # X = Dense(128)(X)
    # X = LeakyReLU(0.01)(X)
    X = Dropout(0.5)(X)
    output = Dense(19, activation='softmax', name='behaviour',
                   # kernel_regularizer=regularizers.l2(1e-5)
                   )(X)
    side = CenterLossLayer(alpha=0.5, name='centerlosslayer')([X, input2])
    return Model([input1, input2], [output, side])
    # output1 = Dense(19, activation='softmax', name='behaviour')(X)
    # output2 = Dense(7, activation='softmax', name='movement')(X)
    # return Model([input], [output1, output2])