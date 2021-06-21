import tensorflow as tf
import numpy as np
from scipy import optimize
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

def wasserstein_distance(p, q, C):
    """Wasserstein距离计算方法，
    p.shape=(m,)
    q.shape=(n,)
    C.shape=(m,n)
    p q满足归一性概率化
    """
    p = np.array(p)
    q = np.array(q)
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(C)
        A[i,:] = 1.0
        A_eq.append(A.reshape((-1,)))
    for i in range(len(q)):
        A = np.zeros_like(C)
        A[:,i] = 1.0
        A_eq.append(A.reshape((-1,)))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q], axis=0)
    C = np.array(C).reshape((-1,))
    return optimize.linprog(
        c=C,
        A_eq=A_eq[:-1],
        b_eq=b_eq[:-1],
        method="interior-point",
        options={"cholesky":False, "sym_pos":True}
    ).fun
