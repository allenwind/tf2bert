import tensorflow as tf
import numpy as np

def manhattan(x, y):
    # L1
    return tf.reduce_sum(tf.abs(x - y))

def euclidean(x, y):
    # L2
    return tf.sqrt(tf.reduce_sum((x - y) ** 2))

def minkowski(x, y, p):
    # Lp
    return tf.reduce_sum(tf.abs(x - y) ** p) ** (1 / p)

def chebyshev(x, y):
    # L-infinity
    return tf.reduce_max(tf.abs(x - y))

def hamming(x, y):
    return tf.reduce_sum(~tf.equal(x, y)) / tf.size(x)

def py_minkowski(x, y, p=2):
    return np.sum(np.abs(x-y)**p) ** (1 / p)
