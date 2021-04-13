import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import initializers

# 常见的Pooling层

class MaskedGlobalMaxPooling1D(Layer):
    
    def __init__(self, return_scores=False, **kwargs):
        super(MaskedGlobalMaxPooling1D, self).__init__(**kwargs)
        self.return_scores = return_scores

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        x = inputs
        x = x - (1 - mask) * 1e12 # 用一个大的负数mask
        x = tf.reduce_max(x, axis=1, keepdims=True)
        if self.return_scores:
            ws = tf.where(inputs == x, x, 0.0)
            ws = tf.reduce_sum(ws, axis=2)
            x = tf.squeeze(x, axis=1)
            return x, ws
        x = tf.squeeze(x, axis=1)
        return x

class MaskedGlobalAveragePooling1D(Layer):
    
    def __init__(self, return_scores=False, **kwargs):
        super(MaskedGlobalAveragePooling1D, self).__init__(**kwargs)
        self.return_scores = return_scores

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        x = inputs
        x = x * mask
        x = tf.reduce_sum(x, axis=1)
        x =  x / tf.reduce_sum(mask, axis=1)
        if self.return_scores:
            ws = tf.square(inputs - tf.expand_dims(x, axis=1))
            ws = tf.reduce_mean(ws, axis=2)
            ws = ws + (1 - mask) * 1e12
            ws = 1 / ws
            return x, ws
        return x

class AttentionPooling1D(Layer):

    def __init__(self, hdims, return_scores=False, kernel_initializer="glorot_uniform", **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.hdims = hdims
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.return_scores = return_scores
        self.supports_masking = False

    def build(self, input_shape):
        self.k_dense = Dense(
            units=self.hdims,
            kernel_initializer=self.kernel_initializer,
            activation="tanh",
            use_bias=False,
        )
        self.o_dense = Dense(
            units=1,
            use_bias=False
        )

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        # 计算每个 time steps 权重
        w = self.k_dense(inputs)
        w = self.o_dense(w)
        # 处理 mask
        w = w - (1 - mask) * 1e12
        # 权重归一化
        w = tf.math.softmax(w, axis=1)
        # 加权平均
        x = tf.reduce_sum(w * inputs, axis=1)
        if self.return_scores:
            return x, w
        return x

class MultiHeadAttentionPooling1D(Layer):

    def __init__(
        self,
        hdims,
        heads,
        return_scores=False,
        kernel_initializer="glorot_uniform",
        **kwargs
    ):
        super(MultiHeadAttentionPooling1D, self).__init__(**kwargs)
        self.hdims = hdims
        self.heads = heads
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.return_scores = return_scores
        self.supports_masking = False

    def build(self, input_shape):
        """k_dense可以理解长特征维度的变换，不参与多头相关的操作
        因此这里参数不变。当然也可以参与多头操作，但是参数会变多。"""
        self.k_dense = Dense(
            units=self.hdims,
            kernel_initializer=self.kernel_initializer,
            activation="tanh",
            use_bias=False,
        )
        self.o_dense = Dense(
            units=self.heads,
            use_bias=False
        )

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        x0 = inputs
        # 计算每个 time steps 权重
        w = self.k_dense(inputs)
        w = self.o_dense(w)
        # 处理 mask
        w = w - (1 - mask) * 1e12
        # 权重归一化
        # (batch_size, seqlen, heads)
        w = tf.math.softmax(w, axis=1) # 有mask位置对应的权重变为很小的值
        # 加权平均
        # （batch_size, seqlen, heads, 1) * (batch_size, seqlen, 1, hdims) 
        # 这里直接对原始输入进行加权平均，因此要考虑维度要一致
        x = tf.reduce_sum(
            tf.expand_dims(w, axis=-1) * tf.expand_dims(x0, axis=2),
            axis=1
        )
        x = tf.reshape(x, (-1, self.heads * self.hdims))
        if self.return_scores:
            return x, w
        return x

class MaskedMinVariancePooling(Layer):
    """最小方差加权平均，Inverse-variance weighting
    等价于正太分布的最小熵加权平均"""

    def __init__(self, return_scores=False, **kwargs):
        super(MaskedMinVariancePooling, self).__init__(**kwargs)
        self.return_scores = return_scores

    def build(self, input_shape):
        d = tf.cast(input_shape[2], tf.float32)
        self.alpha = 1 / (d - 1)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        mu = tf.reduce_mean(inputs, axis=2, keepdims=True) # 均值
        var = self.alpha * tf.reduce_sum(tf.square(inputs - mu), axis=2, keepdims=True) # 方差的无偏估计
        var = var + (1 - mask) * 1e12 # 倒数的mask处理
        ivar = 1 / var
        w = ivar / tf.reduce_sum(ivar, axis=1, keepdims=True)
        x = tf.reduce_sum(inputs * w * mask, axis=1)
        if self.return_scores:
            return x, w
        return x
