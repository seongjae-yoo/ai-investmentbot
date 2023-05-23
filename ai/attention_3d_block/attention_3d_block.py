'''
참고한 사이트
https://github.com/PatientEz/CNN-BiLSTM-Attention-Time-Series-Prediction_Keras/blob/master/Main.py
'''
from keras.layers import Input, Dense, LSTM, merge ,Conv1D,Dropout,Bidirectional,Multiply 
from keras.models import Model

##### 05-23 Add
from tensorflow.keras.layers import Permute, RepeatVector, Lambda, Add
from tensorflow.keras.backend import mean, int_shape
from tensorflow.keras.activations import tanh


#from attention_utils import get_activations
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

import pandas as pd
import numpy as np

import keras.backend as K
from ..SPPModel import *




SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    #a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

# 注意力机制的另一种写法 适合上述报错使用 来源:https://blog.csdn.net/uhauha2929/article/details/80733255
def attention_3d_block2(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
   
    a = Permute((2, 1))(inputs) # 주어진 패턴에 따라서 인풋의 차원을 치환합니다
    a = Dense(time_steps, activation='softmax')(a) # 원본
   
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


###################################################################################################################3

def attention_3d_block3(inputs, single_attention_vector=False):
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]

    a = Permute((2, 1))(inputs)
    query = Dense(input_dim, activation='linear')(a)  # Query (no activation)
    value = Dense(input_dim, activation='linear')(a)  # Value (no activation)

    query_value_dot = Multiply()([query, value])
    a = Dense(time_steps, activation='softmax')(query_value_dot)  # Multiplicative attention (Luong's style)

    if single_attention_vector:
        a = Lambda(lambda x: mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])

    return output_attention_mul

#############################################################################################################################3


def attention_3d_block4(inputs, single_attention_vector=False):
    time_steps = int_shape(inputs)[1]
    input_dim = int_shape(inputs)[2]

    a = Permute((2, 1))(inputs)

    query = Dense(input_dim, activation='linear')(a)  # Query (no activation)
    value = Dense(input_dim, activation='linear')(a)  # Value (no activation)

    query_value_sum = Add()([query, value])
    a = Dense(time_steps, activation='tanh')(query_value_sum)  # Additive attention (Bahdanau's style) # tf.keras.activations.tanh
    a = Dense(time_steps, activation='softmax')(a)

    if single_attention_vector:
        a = Lambda(lambda x: mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])

    return output_attention_mul

#########################################################################################################################################
    
class TransformedAttention(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super(TransformedAttention, self).__init__(**kwargs)
        self.dim = dim
        self.W_q = self.add_weight(shape=(dim, dim), initializer='random_normal')
        self.W_k = self.add_weight(shape=(dim, dim), initializer='random_normal')

    def call(self, query, key, value):
        q = tf.matmul(query, self.W_q)
        k = tf.matmul(key, self.W_k)
        attention = tf.nn.softmax(tf.matmul(q, k, transpose_b=True))
        return tf.matmul(attention, value)
    
    def get_config(self):
        config = super(TransformedAttention, self).get_config()
        config.update({"dim": self.dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

###################################################################################3

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def call(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        return tf.matmul(attention_weights, value)