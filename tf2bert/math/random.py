import numpy as np
import tensorflow as tf

def reduce_moments():
    pass

def reduce_variance(x, axis=None, keepdims=False, name="reduce_variance"):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    diff = x - m
    sdiff = tf.square(diff)
    return tf.reduce_mean(sdiff, axis=axis, keepdims=keepdims)

def inverse_variance_weighted_sum(vs, return_weights=False):
    """逆方差加权平均"""
    mu = np.mean(vs, axis=1, keepdims=True)
    var = 1 / (vs.shape[1] - 1) * np.sum(np.square(vs - mu), axis=1, keepdims=True)
    ivar = 1 / var
    w = ivar / np.sum(ivar, axis=0, keepdims=True)
    s = np.sum(vs * w, axis=0)
    if return_weights:
        return s, w
    return s

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = np.cumsum(np.random.normal(size=(10, 512)), axis=-1)
    y, w = inverse_variance_weighted_sum(x, return_weights=True)
    plt.plot(x.T, color="blue")
    plt.plot(y, color="red")
    print(w)
    plt.show()
