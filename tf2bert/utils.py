import time
import os
import collections
import pkg_resources
from functools import wraps
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

__all__ = ["list_layer_viewers", "list_all_weights", "plot_learning_curve"]

def paddle_to_checkpoint(checkpoint):
    pass

bert_mapping = collections.OrderedDict([
    ("layer.", "layer_"),
    ("word_embeddings.weight", "word_embeddings"),
    ("position_embeddings.weight", "position_embeddings"),
    ("token_type_embeddings.weight", "token_type_embeddings"),
    (".", "/"),
    ("LayerNorm/weight", "LayerNorm/gamma"),
    ("LayerNorm/bias", "LayerNorm/beta"),
    ("weight", "kernel"),
    ("cls/predictions/bias", "cls/predictions/output_bias"),
    ("cls/seq_relationship/kernel", "cls/seq_relationship/output_weights"),
    ("cls/seq_relationship/bias", "cls/seq_relationship/output_bias")
])

transpose = ("dense.weight", "attention.self.query", "attention.self.key", "attention.self.value")

def pattern_replace(name, mapping=bert_mapping):
    for old, new in mapping.items():
        name = name.replace(old, new)
    return name

def torch_to_checkpoint(torch_file, tf_file=None):
    try:
        import torch
    except ImportError as err:
        print("install pytorch from: https://pytorch.org/get-started/locally/")
        raise err

    weights = torch.load(torch_file, map_location="cpu")
    # kwargs = {}
    # for name, weight in weights.items():
    #     weight = weight.numpy()
    #     if any([x in name for x in transpose]):
    #         weight = weight.T
    #     name = pattern_replace(name)
    #     variable = tf.Variable(weight, name=name)
    #     kwargs[name] = variable

    # ck = tf.train.Checkpoint(**kwargs)
    # if tf_file is not None:
    #     ck.save(tf_file)
    # return ck
    with tf.Graph().as_default():
        for name, weight in weights.items():
            weight = weight.numpy()
            if any(x in name for x in transpose):
                weight = weight.T
            name = pattern_replace(name)
            variable = tf.Variable(weight, name=name)

        # TODO:remove compat code
        with tf.compat.v1.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            saver.save(session, tf_file, write_meta_graph=False)

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

def list_variables(checkpoint, string=False):
    """列出checkpoint的所有variables"""
    ck = tf.train.load_checkpoint(checkpoint)
    if string:
        return ck.debug_string().decode("utf-8")
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
