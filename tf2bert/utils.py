import time
import os
import pkg_resources
from functools import wraps
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

__all__ = ["list_layer_viewers", "list_all_weights", "plot_learning_curve"]

def disable_gpu():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, "elapsed time {:.3f}s".format(end-start))
        return result
    return wrapper

def list_variables(checkpoint):
    """列出checkpoint的所有variables"""
    ck = tf.train.load_checkpoint(checkpoint)
    return ck.get_variable_to_shape_map()

def list_dependencies(name):
    """根据包名列出其依赖"""
    package = pkg_resources.working_set.by_key[name]
    return [str(r) for r in package.requires()]

def one_hot(indices, num_classes):
    vec = np.zeros((num_classes,), dtype=np.int64)
    vec[indices] = 1
    return vec

def ndim(x):
    dims = x.shape._dims
    if dims is not None:
        return len(dims)
    return None

def int_shape(x):
    return tuple(x.shape)

def list_layer_viewers(model):
    # 建立每层的可视化模型
    layers = model.layers
    inputs = layers[0]
    viewers = []
    for o in layers[1:]:
        viewers.append(tf.keras.models.Model(inputs, o))
    return viewers

def list_all_weights(model):
    # 列出每层的权重
    ws = [] # weights
    bs = [] # biases
    for i in range(1, len(model.layers)):
        weights, biases = model.layers[i].get_weights()
        ws.append(weights)
        bs.append(biases)
    return ws, bs

def plot_learning_curve(model, show=True):
    # 绘制神经网络的学习曲线和监控指标
    if not isinstance(model, tf.keras.models.Model):
        raise Exception("only support tf.keras.models.Model")

    for label, hs in model.history.history.items():
        plt.plot(hs, label=label)

    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.legend(loc="upper right")

    if show:
        plt.show()

class Timer:
    """简单的计时器"""

    def __init__(self, counter=time.perf_counter):
        self.elapsed = 0.0
        self._counter = counter
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError("Already started")
        self._start = self._counter()

    def stop(self):
        if self._start is None:
            raise RuntimeError("Not started")
        end = self._counter()
        self.elapsed += end - self._start
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    def __repr__(self):
        return "<Timer(elapsed={})>".format(self.elapsed)

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
