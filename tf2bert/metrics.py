import tensorflow as tf


class ChunkingPRF1(tf.keras.metrics.Metric):
    """NER的precision、recall、F1指标"""

    def __init__(self, from_potentials=False):
        self.precision = self.add_weight(name='precision', initializer='zeros')
        self.recall = self.add_weight(name='recall', initializer='zeros')
        self.f1 = self.add_weight(name='f1', initializer='zeros')

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
