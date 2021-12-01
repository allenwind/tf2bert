import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 混合精度
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import numpy as np
import pandas as pd
import os
import warnings
import collections

import dataset
import tokenizer
from layers import MultiHeadAttention, AttentionPooling1D, gelu, BidirectionalConcatenate, VIB

import tfx

# TODO 正序、逆序拼接
# TODO BERT char features
# TODO 用tf.data改写引入更多随机性
# TODO 引入半监督训练才初始化Embedding
# 添加句向量到词向量的loss，参考word2ve skip-gram
# 加入对抗训练

# 基于CNN+AttentionPooling1D的文本匹配
# lcqmc：val_accuracy: 0.9007 by validation_split
# lcqmc: val_accuracy: 0.8142, test_accuracy: 0.8182
# 0.8449
# afqmc
# https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16126/16099

class CharWordMixEmbedding:
    """字词混合Embedding"""
    pass

# X1, X2, y = dataset.load_afqmc()
X1, X2, y = dataset.load_lcqmc()

ctokenizer = tokenizer.CharTokenizer(min_freq=8)
# 构建全局词表
ctokenizer.fit(X1 + X2)

X1 = ctokenizer.transform(X1)
X2 = ctokenizer.transform(X2)

maxlen = 64
hdims = 128
vocab_size = ctokenizer.vocab_size

X1 = tf.keras.preprocessing.sequence.pad_sequences(X1, maxlen=maxlen)
X2 = tf.keras.preprocessing.sequence.pad_sequences(X2, maxlen=maxlen)
y = tf.keras.utils.to_categorical(y)

x1_input = Input(shape=(maxlen,))
x2_input = Input(shape=(maxlen,))

# 计算全局mask
x1_mask = Lambda(lambda x: tf.not_equal(x, 0))(x1_input)
x2_mask = Lambda(lambda x: tf.not_equal(x, 0))(x2_input)

embedding = Embedding(vocab_size, hdims, embeddings_initializer="normal", mask_zero=True)
# biconcat = BidirectionalConcatenate()
# attn = MultiHeadAttention(hdims, num_heads=8)
layernom = LayerNormalization()

x1 = embedding(x1_input)
# x1 = attn([x1, x1, x1], x1_mask)
x1 = layernom(x1) # 收敛更快，且更稳定
# x1 = biconcat(x1) # +up
x1 = Dropout(0.05)(x1)

x2 = embedding(x2_input)
# x2 = attn([x2, x2, x2], x2_mask)
x2 = layernom(x2)
# x2 = biconcat(x2)
x2 = Dropout(0.05)(x2)

# 位置信息
# pe = tfx.layers.embeddings.SinCosPositionEmbedding(maxlen, hdims, trainable=True)
# x1 += pe(x1)
# x2 += pe(x2)

class GatedConv1D(tf.keras.layers.Layer):
    
    def __init__(self, hdims, kernel_size=2, **kwargs):
        super(GatedConv1D, self).__init__(**kwargs)
        self.hdims = hdims
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = Conv1D(self.hdims * 2, self.kernel_size, padding="same", activation=gelu)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        x0 = inputs
        x = self.conv1(x0 * mask)
        x, g = tf.split(x, num_or_size_splits=2, axis=-1)
        g = tf.math.sigmoid(g)
        return (x0 * (1 - g) + x * g) * mask

# conv1 = Conv1D(filters=hdims, kernel_size=2, padding="same", activation=gelu)
# conv2 = Conv1D(filters=hdims, kernel_size=3, padding="same", activation=gelu)
# conv3 = Conv1D(filters=hdims, kernel_size=4, padding="same", activation=gelu)

# x1 = conv1(x1) + x1
# x1 = conv2(x1) + x1
# x1 = conv3(x1) + x1

# x2 = conv1(x2) + x2
# x2 = conv2(x2) + x2
# x2 = conv3(x2) + x2

conv1 = GatedConv1D(hdims, 2)
conv2 = GatedConv1D(hdims, 3)
conv3 = GatedConv1D(hdims, 4)

x1 = conv1(x1)
x1 = conv2(x1)
x1 = conv3(x1)

x2 = conv1(x2)
x2 = conv2(x2)
x2 = conv3(x2)

# attn = MultiHeadAttention(hdims, 4)
# x1 = attn([x1, x1, x1], x1_mask)
# x2 = attn([x2, x2, x2], x2_mask)

pool = AttentionPooling1D(hdims)

x1 = pool(x1, mask=x1_mask)
x2 = pool(x2, mask=x2_mask)


# for VIB
d1 = Dense(hdims)
d2 = Dense(hdims)
vib = VIB(0.1)

z_mean_1 = d1(x1)
z_log_var_1 = d2(x1)
x1 = vib([z_mean_1, z_log_var_1])

z_mean_2 = d1(x2)
z_log_var_2 = d2(x2)
x2 = vib([z_mean_2, z_log_var_2])

# x*y
x3 = Multiply()([x1, x2])
# |x-y|
x4 = Lambda(lambda x: tf.abs(x[0] - x[1]))([x1, x2])

# (x+y)/2

# [x;y]
# [y;x]

x = Concatenate()([x1, x2, x3, x4])
x = Dense(4 * hdims, kernel_regularizer="l2")(x)
x = Dropout(0.3)(x) # 模拟集成
x = gelu(x) # 有一点提升
# x = LeakyReLU(0.2)(x)
outputs = Dense(2, activation="softmax")(x)

model = Model([x1_input, x2_input], outputs)
model.summary()

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[0, 2, 6],
    values=[2*1e-3, 0.8*1e-3, 0.5*1e-3, 1e-4]
)
adam = tf.keras.optimizers.Adam(lr) # lr=0.8*1e-3 设置更小的学习率，避免训练loss震荡
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

# 加载验证集合
# val_X1, val_X2, val_y = dataset.load_lcqmc("dev.txt")
# val_X1 = ctokenizer.transform(val_X1)
# val_X2 = ctokenizer.transform(val_X2)

# val_X1 = tf.keras.preprocessing.sequence.pad_sequences(val_X1, maxlen=maxlen)
# val_X2 = tf.keras.preprocessing.sequence.pad_sequences(val_X2, maxlen=maxlen)
# val_y = tf.keras.utils.to_categorical(val_y)

model.fit([X1, X2], y, shuffle=True, batch_size=32, epochs=20, validation_split=0.1)#validation_data=([val_X1, val_X2], val_y))

# 加载测试集合
# test_X1, test_X2, test_y = dataset.load_lcqmc("test.txt")
# test_X1 = ctokenizer.transform(test_X1)
# test_X2 = ctokenizer.transform(test_X2)

# test_X1 = tf.keras.preprocessing.sequence.pad_sequences(test_X1, maxlen=maxlen)
# test_X2 = tf.keras.preprocessing.sequence.pad_sequences(test_X2, maxlen=maxlen)
# test_y = tf.keras.utils.to_categorical(test_y)
# print(model.evaluate([test_X1, test_X2], test_y))
