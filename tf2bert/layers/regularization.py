import numpy as np
import tensorflow as tf

class RandomChange(tf.keras.layers.Layer):
    """随机替换tokens，可以优化的地方：指定tokens集"""
    
    def __init__(self, num_words, rate=0.3, **kwargs):
        super(RandomChange, self).__init__(**kwargs)
        self.num_words = num_words
        self.rate = rate

    def call(self, inputs, training=None):
        # use tf.float32?
        if training:
            batchs = tf.shape(inputs)[0]
            maxlen = tf.shape(inputs)[-1]
            mask = tf.random.uniform((batchs, maxlen), minval=0, maxval=1)
            mask = tf.cast(mask < self.rate, tf.int32)
            tokens = tf.random.uniform(
                shape=(batchs, maxlen),
                minval=2,
                maxval=self.num_words,
                dtype=tf.int32
            )
            return inputs * (1 - mask) + tokens * mask
        return inputs

class Dropout(tf.keras.layers.Layer):
    """tf.nn.dropout的Keras封装"""

    def __init__(self, rate, noise_shape, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape

    @tf.function
    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=1 - self.rate)
        return inputs
