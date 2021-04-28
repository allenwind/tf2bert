import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import initializers
from tensorflow.keras import activations

class LayerNormalization(tf.keras.layers.Layer):
    """LayerNormalization层归一化的基本实现"""

    def __init__(
        self,
        center=True,
        scale=True,
        epsilon=1e-12, # 1e-3
        hidden_units=None,
        hidden_activation="linear",
        hidden_initializer="glorot_uniform",
        **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)
        self.supports_masking = True

    def build(self, input_shape):
        shape = (input_shape[-1],)
        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer="zeros", name="beta"
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer="ones", name="gamma"
            )

    def call(self, inputs):
        if self.center:
            beta = self.beta
        if self.scale:
            gamma = self.gamma
        outputs = inputs
        if self.center:
            mean = tf.reduce_mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = tf.reduce_mean(tf.square(outputs), axis=-1, keepdims=True)
            std = tf.sqrt(variance + self.epsilon)
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta
        return outputs

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "center": self.center,
            "scale": self.scale,
            "epsilon": self.epsilon,
            "hidden_units": self.hidden_units,
            "hidden_activation": activations.serialize(self.hidden_activation),
            "hidden_initializer": initializers.serialize(self.hidden_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BatchSequenceNormalization(tf.keras.layers.Layer):
    """在一个batch上序列方向即axis=1计算均值和方差然后再标准化，
    用在时间序列相关问题上。"""

    def __init__(
        self,
        epsilon=1e-3,
        center=True,
        scale=True,
        trainable=True,
        **kwargs):
        super(BatchSequenceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon # 避免出现0方差
        self.center = center
        self.scale = scale
        self.trainable = trainable

    def build(self, input_shape):
        shape = (input_shape[-1],)
        # 使用简单的初始化
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer="zeros",
                trainable=self.trainable,
                name="beta"
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer="ones",
                trainable=self.trainable,
                name="gamma"
            )

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1.0
        else:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
        x = inputs
        if self.center:
            # (1, 1, hdims)
            mean = tf.reduce_sum(inputs, axis=[0, 1], keepdims=True) / tf.reduce_sum(mask)
            x = x - mean
        if self.scale:
            variance = tf.reduce_sum(tf.square(x), axis=[0, 1], keepdims=True) / tf.reduce_sum(mask)
            std = tf.sqrt(variance + self.epsilon)
            x = x / std * self.gamma
        if self.center:
            x = x + self.beta
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

class BatchNormalization(tf.keras.layers.Layer):
    """在Batch维度的归一化，通常用在图像上，NLP中使用较少"""

    def __init__(
        self,
        epsilon=1e-3,
        momentum=0.99,
        center=True,
        scale=True,
        trainable=True,
        **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon # 避免出现0方差
        self.momentum = momentum # 用于mean和std的移动平均
        self.center = center
        self.scale = scale
        self.trainable = trainable

    def build(self, input_shape):
        # moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)
        # moving_var = moving_var * momentum + var(batch) * (1 - momentum)
        pass

    def call(self, inputs):
        """训练阶段和推断阶段有所不同"""

        # training:(batch - mean(batch)) / (var(batch) + epsilon) * gamma + beta
        # inference:(batch - moving_mean) / (moving_var + epsilon) * gamma + beta
        pass

class GroupNormalization(tf.keras.layers.Layer):
    """GroupNormalization（特征维度进行分组）"""
    
    def __init__(self, group_axis=(-1,)):
        self.group_axis = group_axis
        self.epsilon = 1e-7

    def build(self, input_shape):
        pass

class InstanceNormalization(tf.keras.layers.Layer):
    """InstanceNormalization（GroupNormalization的特例，分组数为特征数）"""
    pass

class MinMaxScaling1D(tf.keras.layers.Layer):
    
    def __init__(self, params, **kwargs):
        super(MinMaxScaling1D, self).__init__(**kwargs)
        self.params = params

    def build(self, input_shape):
        tf.constant(self.params, shape=(1, 1, len(self.params)))

    def call(self, inputs, mask=None):
        pass
