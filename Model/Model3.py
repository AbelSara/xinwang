from keras.layers import *
from keras.models import Model
# from keras_self_attention import SeqSelfAttention
# from keras.regularizers import l1, l2
from Model.Sub_block import residual_block
def Net():
    input = Input(shape=(60, 8, 1))
    X = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same')(input)

    X = LeakyReLU(0.01)(X)
    #X = Dropout(0.2)(X)
    # X = BatchNormalization()(X)
    # X = Conv2D(filters=128,
    #            kernel_size=(3, 3),
    #            padding='same')(X)
    #
    # X = LeakyReLU(0.01)(X)
    # X = Dropout(0.2)(X)
    X = residual_block(X, input_channels=64, output_channels=128)
    X = Dropout(0.2)(X)
    X = MaxPooling2D((2, 2))(X)
    # X = BatchNormalization()(X)

    X = residual_block(X, input_channels=128, output_channels=256)
    X = Dropout(0.3)(X)
    # X = residual_block(X, input_channels=128, output_channels=128)
    # X = Dropout(0.2)(X)
    X = MaxPooling2D((1, 2))(X)

    # X = residual_block(X, input_channels=128, output_channels=256)
    # X = Dropout(0.2)(X)
    X = residual_block(X, input_channels=256, output_channels=512)
    X = Dropout(0.4)(X)
    # X = BatchNormalization()(X)
    # X = Reshape((-1, 2*512))(X)
    #X = Dropout(0.3)(X)
    #X = Dense(128, activation='relu')(X)
    # X = GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(X)
    # X = GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(X)

    # X = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    #                        kernel_regularizer=l2(1e-4),
    #                        bias_regularizer=l1(1e-4),
    #                        attention_regularizer_weight=1e-4,
    #                        name='Attention')(X)

    X = GlobalAveragePooling2D()(X)
    X = Dropout(0.5)(X)
    X = Dense(19, activation='softmax')(X)
    return Model([input], X)