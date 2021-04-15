import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras.layers import *

# Dense相关的扩展层

class BiasAdd(tf.keras.layers.Layer):
    """tf.nn.bias_add的Keras Layer封装"""

    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.bias = self.add_weight(
            name="bias",
            shape=(output_dim,),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs):
        return tf.nn.bias_add(inputs, self.bias)

class NoisyDense(tf.keras.layers.Layer):
    """在权重矩阵中添加随机噪声，可参看论文
    [Noisy Networks for Explanation](https://arxiv.org/pdf/1706.10295.pdf)"""

class FeedForward(tf.keras.layers.Layer):
    """Transformer中的position-wise feed-forward networks层，
    在此基础上还有很多的变种，这个根据需要扩展。"""

    def __init__(self, units, use_bias=True, activation="relu", kernel_initializer="glorot_uniform", **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        activation = activation if isinstance(activation, list) else [activation]
        self.activation = [activations.get(i) for i in activation]
        self.activation_size = len(self.activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.supports_masking = True

    def build(self, input_shape):
        for i, act in enumerate(self.activation):
            i_dense = Dense(
                units=self.units,
                activation=act,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
            setattr(self, "i{}_dense".format(i), i_dense)

        self.o_dense = Dense(
            units=input_shape[-1],
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    def call(self, inputs):
        x = self.i0_dense(inputs)
        for i in range(1, self.activation_size):
            layer = getattr(self, "i{}_dense".format(i))
            x = x * layer(inputs)
        x = self.o_dense(x)
        return x

    def get_config(self):
        base = super(FeedForward, self).get_config()
        configs = {
            "units": self.units,
            "use_bias": self.use_bias,
            "activation": [activations.serialize(i) for i in self.activation],
            "kernel_initializer": initializers.serialize(self.kernel_initializer)
        }
        return dict(list(base.items()) + list(configs.items()))
