import tensorflow as tf

# 序列标注相关的指标
# 需要注意在NER中，有两类指标，
# 一类是直接根据识别的实体集合计算
# 另外一类是根据实体类别及其位置计算，后者更严格
# 只要知道序列的mask就容易计算了 mask = y_pred._keras_mask


class ChunkingPRF1(tf.keras.metrics.Metric):
    """NER的precision、recall、F1指标"""

    def __init__(self, from_potentials=False):
        self.precision = self.add_weight(name="precision", initializer="zeros")
        self.recall = self.add_weight(name="recall", initializer="zeros")
        self.f1 = self.add_weight(name="f1", initializer="zeros")

    def __call__(self, y_true, y_pred, sample_weight=None):
        pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        pass

class ChunkingRegionPRF1(ChunkingPRF1):
    """CWS的precision、recall、F1指标"""

    def to_region(self):
        pass

class PRF1Score:

    def __init__(self):
        self.reset()

    def reset(self):
        self.num_correct = 0.0
        self.num_pred = 0.0
        self.num_true = 0.0

    def __repr__(self):
        return "P:{:.2%} R:{:.2%} F1:{:.2%}".format(*self.prf)

    @property
    def prf(self):
        num_correct = self.num_correct
        num_pred = self.num_pred
        num_true = self.num_true
        p = num_correct / num_pred if num_pred > 0 else 0.0
        r = num_correct / num_true if num_true > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        return p, r, f

    def __call__(self, y_true, y_pred):
        """y_pred形如("PER", start, end)"""
        y_true = set(y_true)
        y_pred = set(y_pred)
        self.num_correct += len(y_true & y_pred)
        self.num_true += len(y_true)
        self.num_pred += len(y_pred)

def evaluate_prf(y_true, y_pred):
    X = Y = Z = 1e-10
    for R, T in zip(y_pred, y_true):
        R = set(R)
        T = set(T)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    precision = X / Y
    recall = X / Z
    f1 = 2 * X / (Y + Z)
    return precision, recall, f1
