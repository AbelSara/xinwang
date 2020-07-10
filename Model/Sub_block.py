from keras import initializers, regularizers, constraints
from keras.layers import *
import keras.backend as K

def residual_block(input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1):
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """
    if output_channels is None:
        output_channels = input.get_shape()[-1].value
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride)

    #x = BatchNormalization(epsilon=0.001, momentum=0.99)(input)

    x = Conv2D(output_channels, kernel_size, padding='same', strides=stride)(x)
    x = LeakyReLU(0.01)(x)
    x = Dropout(0.3)(x)

    #x = BatchNormalization(epsilon=0.001, momentum=0.99)(x)
    x = Conv2D(output_channels, kernel_size, padding='same', strides=stride)(x)

    if input_channels != output_channels or stride != 1:
        input = Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)

    x = Add()([x, input])
    x = LeakyReLU(0.01)(x)
    return x

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
# class attentionLayer(Layer):
#     def __init__(self, **kwargs):
#         """"
#         Class-wise attention pooling layer
#                 Args:
#                 Attributes:
#             kernel: tensor
#             bias: tensor
#
#         """
#         super(attentionLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         kernel_shape = [1] * len(input_shape)
#         bias_shape = tuple([1] * (len(input_shape) - 1))
#
#         kernel_shape[-1] = input_shape[-1]
#         kernel_shape = tuple(kernel_shape)
#
#         self.kernel = self.add_weight(
#             shape=kernel_shape,
#             initializer=Zeros(),
#             name='%s_kernel' % self.name)
#
#         self.bias = self.add_weight(
#             shape=bias_shape,
#             initializer=Zeros(),
#             name='%s_bias' % self.name)
#
#         super(attentionLayer, self).build(input_shape)
#
#     def call(self, inputs):
#         weights = K.sum(inputs * self.kernel, axis=-1) + self.bias
#         return weights
#
#     def compute_output_shape(self, input_shape):
#         out_shape = []
#         for i in range(len(input_shape) - 1):
#             out_shape += [input_shape[i]]
#         return tuple(out_shape)

# def attention_3d_block(hidden_states):
#     """
#     Many-to-one attention mechanism for Keras.
#     @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
#     @return: 2D tensor with shape (batch_size, 128)
#     @author: felixhao28.
#     """
#     hidden_size = int(hidden_states.shape[2])
#     # Inside dense layer
#     #              hidden_states            dot               W            =>           score_first_part
#     # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
#     # W is the trainable weight matrix of attention Luong's multiplicative style score
#     score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
#     #            score_first_part           dot        last_hidden_state     => attention_weights
#     # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
#     h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
#     score = dot([score_first_part, h_t], [2, 1], name='attention_score')
#     attention_weights = Activation('softmax', name='attention_weight')(score)
#     # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
#     context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
#     pre_activation = concatenate([context_vector, h_t], name='attention_output')
#     attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
#     return attention_vector
#
# class attention(Layer):
#     def __init__(self,**kwargs):
#         super(attention,self).__init__(**kwargs)
#
#     def build(self,input_shape):
#         self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
#         self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
#         super(attention, self).build(input_shape)
#
#     def call(self,x):
#         et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
#         at=K.softmax(et)
#         at=K.expand_dims(at,axis=-1)
#         output=x*at
#         return K.sum(output,axis=1)
#
#     def compute_output_shape(self,input_shape):
#         return (input_shape[0],input_shape[-1])
#
#     def get_config(self):
#         return super(attention,self).get_config()
#
# def attention_block(input, input_channels=None, output_channels=None, encoder_depth=1):
#     """
#     attention block
#     https://arxiv.org/abs/1704.06904
#     """
#
#     p = 1
#     t = 2
#     r = 1
#
#     if input_channels is None:
#         input_channels = input.get_shape()[-1].value
#     if output_channels is None:
#         output_channels = input_channels
#
#     # First Residual Block
#     for i in range(p):
#         input = residual_block(input)
#
#     # Trunc Branch
#     output_trunk = input
#     for i in range(t):
#         output_trunk = residual_block(output_trunk)
#
#     # Soft Mask Branch
#
#     ## encoder
#     ### first down sampling
#     output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
#     for i in range(r):
#         output_soft_mask = residual_block(output_soft_mask)
#
#     skip_connections = []
#     for i in range(encoder_depth - 1):
#
#         ## skip connections
#         output_skip_connection = residual_block(output_soft_mask)
#         skip_connections.append(output_skip_connection)
#         # print ('skip shape:', output_skip_connection.get_shape())
#
#         ## down sampling
#         output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
#         for _ in range(r):
#             output_soft_mask = residual_block(output_soft_mask)
#
#             ## decoder
#     skip_connections = list(reversed(skip_connections))
#     for i in range(encoder_depth - 1):
#         ## upsampling
#         for _ in range(r):
#             output_soft_mask = residual_block(output_soft_mask)
#         output_soft_mask = UpSampling2D()(output_soft_mask)
#         ## skip connections
#         output_soft_mask = Add()([output_soft_mask, skip_connections[i]])
#
#     ### last upsampling
#     for i in range(r):
#         output_soft_mask = residual_block(output_soft_mask)
#     output_soft_mask = UpSampling2D()(output_soft_mask)
#
#     ## Output
#     output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
#     output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
#     output_soft_mask = Activation('sigmoid')(output_soft_mask)
#
#     # Attention: (1 + output_soft_mask) * output_trunk
#     output = Lambda(lambda x: x + 1)(output_soft_mask)
#     output = Multiply()([output, output_trunk])  #
#
#     # Last Residual Block
#     for i in range(p):
#         output = residual_block(output)
#
#     return output