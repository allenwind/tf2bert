import tensorflow as tf
from tensorflow.keras.layers import *

class MaskedConv1D(tf.keras.layers.Conv1D):
    """不让Conv1D看到mask数据"""

    def __init__(self, **kwargs):
        super(MaskedConv1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            # (batch_size, seq_len, 1)
            mask = tf.expand_dims(mask, axis=-1)
            inputs = inputs * mask
        return super(MaskedConv1D, self).call(inputs)

class ResidualGatedConv1D(tf.keras.layers.Layer):
    """残差门（膨胀）卷积"""

    def __init__(self, filters, kernel_size=2, dilation_rate=1, **kwargs):
        super(ResidualGatedConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.supports_masking = True

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(1,),
            initializer="zeros",
            trainable=True
        )
        self.conv1d = Conv1D(
            filters=self.filters * 2,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding="same",
        )
        self.layernorm = LayerNormalization()
        if self.filters != input_shape[-1]:
            self.o_dense = Dense(self.filters, use_bias=False)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            # (batch_size, seq_len, 1)
            mask = tf.expand_dims(mask, axis=-1)
        else:
            mask = 1.0

        inputs = inputs * mask
        x = self.conv1d(inputs)
        # tf.split
        o1 = x[:, :, self.filters:]
        o2 = x[:, :, :self.filters] 
        x = o1 * tf.sigmoid(o2)
        x = self.layernorm(x)

        if hasattr(self, "o_dense"):
            inputs = self.o_dense(inputs)
        return inputs + self.alpha * x

    def compute_output_shape(self, input_shape):
        shape = self.conv1d.compute_output_shape(input_shape)
        return (shape[0], shape[1], self.filters)
