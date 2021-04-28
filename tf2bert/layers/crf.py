import tensorflow as tf
import tensorflow_addons as tfa

# CRF的简单实现，依赖tensorflow_addons.text中的相关函数
# tf2bert/tests中有两个本CRF实现的例子
# TODO: 去掉对tensorflow_addons的依赖

class CRF(tf.keras.layers.Layer):
    """CRF的实现，包括trans矩阵和viterbi解码"""

    def __init__(self, lr_multiplier=1, trans_mask=None, trans_initializer="glorot_uniform", trainable=True, **kwargs):
        super(CRF, self).__init__(**kwargs)
        # 设置分层学习率
        self.lr_multiplier = lr_multiplier
        # trans特征转移矩阵的mask
        self.trans_mask = trans_mask
        if isinstance(trans_initializer, str):
            trans_initializer = tf.keras.initializers.get(trans_initializer)
        self.trans_initializer = trans_initializer
        self.trainable = trainable
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 3
        units = input_shape[-1]
        self._trans = self.add_weight(
            name="trans",
            shape=(units, units),
            dtype=tf.float32,
            initializer=self.trans_initializer,
            trainable=self.trainable
        )
        if self.lr_multiplier != 1:
            self._trans.assign(self._trans / self.lr_multiplier)

    @property
    def trans(self):
        """trans并不显式定义转移概率，而是转移特征，
        因此数值并不能反映概率大小。但是，相对大小还是
        具有意义的。"""
        if self.lr_multiplier != 1:
            return self.lr_multiplier * self._trans
        return self._trans

    def call(self, inputs, mask=None):
        # 必须要有相应的mask传入
        # 传入方法：
        # 1.手动计算并传入
        # 2.设置Masking层
        # 3.Embedding层参数设置mask_zero=True
        assert mask is not None
        lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
        # CRF的解码，在篱笆网络（Lattice）上的动态规划，即viterbi算法
        viterbi_tags, _ = tfa.text.crf_decode(inputs, self.trans, lengths)
        # (bs, seq_len), (bs, seq_len, units), (bs,), (units, units)
        return viterbi_tags, inputs, lengths, self.trans

    def compute_mask(self, inputs, mask=None):
        return None

class CRFModel(tf.keras.Model):
    """把CRFloss包装成模型，容易扩展各种loss以及复杂的操作。"""

    def __init__(self, base, return_potentials=False, **kwargs):
        super(CRFModel, self).__init__(**kwargs)
        self.base = base
        self.return_potentials = return_potentials
        self.accuracy_fn = tf.keras.metrics.Accuracy(name="accuracy")

    def call(self, inputs):
        return self.base(inputs)

    def summary(self):
        self.base.summary()

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            viterbi_tags, lengths, crf_loss = self.compute_loss(
                x, y, sample_weight, training=True
            )
        grads = tape.gradient(crf_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        mask = tf.sequence_mask(lengths, y.shape[1])
        self.accuracy_fn.update_state(y, viterbi_tags, mask)
        results = {"crf_loss": crf_loss, "accuracy": self.accuracy_fn.result()}
        return results

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        viterbi_tags, lengths, crf_loss = self.compute_loss(
            x, y, sample_weight, training=False
        )
        mask = tf.sequence_mask(lengths, y.shape[1])
        self.accuracy_fn.update_state(y, viterbi_tags, mask)
        results = {"crf_loss": crf_loss, "accuracy": self.accuracy_fn.result()}
        return results

    def predict_step(self, data):
        # 预测阶段，模型只返回viterbi tags即可
        x, *_ = tf.keras.utils.unpack_x_y_sample_weight(data)
        viterbi_tags, potentials, *_ = self(x, training=False)
        if self.return_potentials:
            return viterbi_tags, potentials
        return viterbi_tags

    def compute_loss(self, x, y, sample_weight, training):
        viterbi_tags, potentials, lengths, trans = self(x, training=training)
        # 这里计算CRF的损失，包括归一化因子。这里涉及到递归计算。
        crf_loss, _ = tfa.text.crf_log_likelihood(potentials, y, lengths, trans)
        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight
        return viterbi_tags, lengths, tf.reduce_mean(-crf_loss)

    def accuracy(self, y_true, y_pred):
        viterbi_tags, potentials, lengths, trans = y_pred
        mask = tf.sequence_mask(lengths, y_true.shape[1])
        return self.accuracy_fn(y_true, viterbi_tags, mask)

class CRFWrapper(tf.keras.layers.Wrapper):
    # 参考 WeightNormalization
    pass
