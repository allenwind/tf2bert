import tensorflow as tf
import numpy as np
import scipy.spatial as spatial

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

def mahalanobis_distance(obs, X, center="zero"):
    # mahalanobis distance of obs to X
    # wiki:
    # https://en.wikipedia.org/wiki/Mahalanobis_distance

    # 计算协方差矩阵
    cov = np.cov(X.T)

    # 计算数据集的中心
    if center == "zero":
        center = np.zeros(cov.shape[1])
    else:
        center = np.mean(X, axis=0)

    # 矩阵的逆不一定存在，这里使用矩阵的伪逆
    icov = np.linalg.pinv(cov)
    # 计算 obs 到 center 的 Mahalanobis distance
    d = spatial.distance.mahalanobis(obs, center, icov)
    return d
