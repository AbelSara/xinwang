from keras.layers import *
from keras.models import Model
from keras import regularizers
# from keras_transformer.position import AddPositionalEncoding
# from keras_transformer.attention import MultiHeadSelfAttention
from keras_self_attention import SeqSelfAttention
def Net():
    input = Input(shape=(60, 8, 1))
    X = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               #kernel_regularizer=regularizers.l2(0.0005)
               )(input)
    X = LeakyReLU(0.01)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)

    X = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               #kernel_regularizer=regularizers.l2(0.0005)
               )(X)
    X = LeakyReLU(0.01)(X)
    X = MaxPooling2D((2, 2))(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)

    X = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding='same',
               #kernel_regularizer=regularizers.l2(0.0005)
               )(X)
    X = LeakyReLU(0.01)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)

    X = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding='same',
               #kernel_regularizer=regularizers.l2(0.0005)
               )(X)
    X = LeakyReLU(0.01)(X)
    X = MaxPooling2D((2, 2))(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)

    X = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding='same',
               #kernel_regularizer=regularizers.l2(0.0005)
               )(X)
    X = LeakyReLU(0.01)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    X = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding='same',
               # kernel_regularizer=regularizers.l2(0.0005)
               )(X)
    X = LeakyReLU(0.01)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    X = MaxPooling2D((1, 2))(X)
    X = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding='same',
               #kernel_regularizer=regularizers.l2(0.0005)
               )(X)
    X = LeakyReLU(0.01)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    X = Reshape((-1, 512))(X)
    #X = Dropout(0.3)(X)
    #X = Dense(128, activation='relu')(X)
    X = GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(X)
    X = GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(X)

    X = GlobalAveragePooling1D()(X)
    #X2 = GlobalMaxPooling1D()(X)
    #X = concatenate([X1, X2])
    X = Dropout(0.5)(X)
    X = Dense(19, activation='softmax')(X)
    return Model([input], X)
    # output1 = Dense(19, activation='softmax', name='behaviour')(X)
    # output2 = Dense(7, activation='softmax', name='movement')(X)
    # return Model([input], [output1, output2])