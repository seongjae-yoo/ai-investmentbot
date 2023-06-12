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
'''
Let's define this process mathematically:

Given a 3-dimensional input tensor X ∈ ℝ^(b×t×d), where b, t, and d are the batch size, time steps, and input dimension respectively.

The attention mechanism can be defined as follows:

Permute the dimensions of the input tensor:

A = permute(X), such that A ∈ ℝ^(b×d×t)

Apply a dense layer with 'softmax' activation:

B = softmax(W_s * A + b_s), where W_s ∈ ℝ^(d×t) and b_s ∈ ℝ^t are the weight matrix and bias term of the softmax layer respectively. "*" denotes matrix multiplication, and softmax is applied along the second dimension. As a result, B ∈ ℝ^(b×d×t).

Permute the dimensions back to the original order:

C = permute(B), such that C ∈ ℝ^(b×t×d)

Perform element-wise multiplication of the input tensor and the attention weights:

Y = X * C, where "*" denotes element-wise multiplication. Y ∈ ℝ^(b×t×d) is the output tensor with the same shape as the input tensor X, but with the values scaled according to the attention weights.

So, in a compact notation, the operation performed by the attention_3d_block2 function can be represented as:

Y = X * permute(softmax(W_s * permute(X) + b_s))

'''


def attention_3d_block2(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim) (0,1,2)
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
    
#######################################################################################3

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim

        # Adjust key_dim to be divisible by num_heads
        self.key_dim = key_dim // num_heads * num_heads

        self.wq = tf.keras.layers.Dense(self.key_dim)
        self.wk = tf.keras.layers.Dense(self.key_dim)
        self.wv = tf.keras.layers.Dense(self.key_dim)
        self.dense = tf.keras.layers.Dense(self.key_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim // self.num_heads))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)

        batch_size = tf.shape(q)[0]

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits /= tf.math.sqrt(tf.cast(self.key_dim // self.num_heads, tf.float32))

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_output = tf.matmul(attention_weights, v)

        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.key_dim))

        output = self.dense(attention_output)
        return output

###################################################################################################################



class TransformerBlock_MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dropout_rate=0.1):
        super(TransformerBlock_MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits /= tf.math.sqrt(tf.cast(self.depth, tf.float32))
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = tf.matmul(attention_weights, v)
        
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        
        output = self.dense(attention_output)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.2):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.multihead_attention = TransformerBlock_MultiHeadAttention(
            num_heads=num_heads,
            d_model=d_model,
            dropout_rate=dropout_rate
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, mask=None):
        q = inputs
        k = inputs
        v = inputs

        attn_output = self.multihead_attention(q, k, v, mask=mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
####################################################################################################################################


#########################################################################################################

class TransformerBlock_version2(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock_version2, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)