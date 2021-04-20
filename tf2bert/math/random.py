import tensorflow as tf

def reduce_moments():
    pass

def reduce_variance(x, axis=None, keepdims=False, name="reduce_variance"):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    diff = x - m
    sdiff = tf.square(diff)
    return tf.reduce_mean(sdiff, axis=axis, keepdims=keepdims)
