import numpy as np
import tensorflow as tf

# 默认使用erf的实现
gelu_approximate = False

def gelu_erf(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

def gelu_tanh(x):
    """https://arxiv.org/abs/1606.08415"""
    cdf = 0.5 * (1.0 + tf.tanh((tf.sqrt(2 / 3.141592653589793) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

gelu = gelu_tanh if gelu_approximate else gelu_erf

def leaky_relu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha=alpha)

def mish(x):
    """[Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)
    """
    return x * tf.math.tanh(tf.math.softplus(x))

def softmax(x, axis=-1):
    """Tensorflow实现的softmax"""
    x = x - tf.reduce_max(x, axis, keepdims=True)
    x = tf.exp(x)
    return x / tf.reduce_sum(x, axis, keepdims=True)

def softmax_np(x, axis=-1):
    """Numpy实现的softmax"""
    x = x - x.max(axis=axis, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)


tf.keras.utils.get_custom_objects()["gelu"] = gelu
tf.keras.utils.get_custom_objects()["gelu_erf"] = gelu_erf
tf.keras.utils.get_custom_objects()["gelu_tanh"] = gelu_tanh
tf.keras.utils.get_custom_objects()["leaky_relu"] = leaky_relu
tf.keras.utils.get_custom_objects()["mish"] = mish
