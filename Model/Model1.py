from keras.layers import *
from keras.models import Model
def Net():
    input = Input(shape=(60, 8, 1))
    X = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same')(input)

    X = LeakyReLU(0.01)(X)
    X = Dropout(0.2)(X)
    # X = BatchNormalization()(X)

    X = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same')(X)

    X = LeakyReLU(0.01)(X)
    X = Dropout(0.2)(X)
    X = MaxPooling2D((2, 2))(X)
    # X = BatchNormalization()(X)

    X = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding='same')(X)
    # X = BatchNormalization()(X)
    X = LeakyReLU(0.01)(X)
    X = Dropout(0.2)(X)
    X = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding='same')(X)

    X = LeakyReLU(0.01)(X)
    X = Dropout(0.2)(X)
    X = MaxPooling2D((1, 2))(X)

    X = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding='same')(X)
    # X = BatchNormalization()(X)
    X = LeakyReLU(0.01)(X)
    X = Dropout(0.2)(X)
    X = Conv2D(filters=512,
               kernel_size=(3, 3),
               padding='same')(X)

    X = LeakyReLU(0.01)(X)
    X = Dropout(0.2)(X)
    # X = BatchNormalization()(X)
    X = Reshape((-1, 2*512))(X)
    #X = Dropout(0.3)(X)
    #X = Dense(128, activation='relu')(X)
    X = GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(X)
    #X = GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, inner_init='orthogonal')(X)

    X = GlobalAveragePooling1D()(X)
    X = Dropout(0.3)(X)
    X = Dense(19, activation='softmax')(X)
    return Model([input], X)