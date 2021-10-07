import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import initializers
from tensorflow.keras import activations

# Embedding有关的层如各种PositionEmbedding等

class Embedding(tf.keras.layers.Embedding):
    """为call添加一个参数，可以当做unbias Dense使用，
    用在语言模型的输出计算上。"""

    def call(self, inputs, mode="embedding"):
        assert mode in ("embedding", "dot")
        if mode == "dot":
            return tf.matmul(inputs, self.embeddings, transpose_b=True)
        return super(Embedding, self).call(inputs)

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            return super(Embedding, self).compute_output_shape(input_shape)
        return (input_shape[0], input_shape[1], self.embeddings.shape[0])

class EmbeddingConcatenate(tf.keras.layers.Layer):
    """合并多个Embedding的输出"""
    pass

class GlyphEmbedding(tf.keras.layers.Layer):
    """纯字形Embedding，可参考：
    https://github.com/allenwind/text-glyph-in-NLU
    """
    pass

class CharAlignHybridEmbedding(tf.keras.layers.Layer):
    """字词混合Embedding，以字为基准对齐"""

    def __init__(
        self,
        input_dim,
        output_dim,
        hybridmerge="add",
        max_segment_length=100,
        without_segment_embedding=True,
        embeddings_initializer="uniform",
        **kwargs):
        super(CharAlignHybridEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hybridmerge = hybridmerge if hybridmerge in ("add", "concat") else "add"
        self.max_segment_length = max_segment_length # 词的最大字长度
        self.without_segment_embedding = without_segment_embedding
        self.embeddings_initializer = tf.keras.initializers.get(
            embeddings_initializer
        )

    def build(self, input_shape):
        self.embeddings = Embedding(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            embeddings_initializer=self.embeddings_initializer
        )
        if not self.without_segment_embedding:
            self.segment_embeddings = Embedding(
                input_dim=self.max_segment_length,
                output_dim=self.output_dim,
                embeddings_initializer=self.embeddings_initializer
            )
        if self.hybridmerge == "concat":
            self.o_dense = Dense(self.output_dim)

    def call(self, inputs, mask=None):
        # 字ID，词ID，段ID
        cids, wids, sids = inputs
        if mask is None:
            mask = 1.0
        else:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)

        xc = self.embeddings(cids)
        xw = self.embeddings(wids)
        if not self.without_segment_embedding:
            xs = self.segment_embeddings(sids)
        else:
            xs = 0.0

        if self.hybridmerge == "add":
            u = 2.0 if self.without_segment_embedding else 3.0
            # 叠加后要scale会原来区间
            x = xc + xw + xs / u
        else:
            x = tf.concat([xc, xw, xs], axis=-1)
            # 融合三个Embedding信息
            x = self.o_dense(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)

class SinusoidalPositionEmbedding(tf.keras.layers.Layer):
    """经典的Sin-Cos位置Embedding"""

    def __init__(self, output_dim, merge_mode="add", **kwargs):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        position_ids = tf.range(0, seq_len, dtype=tf.float32)
        position_ids = tf.expand_dims(position_ids, axis=0)
        indices = tf.range(0, self.output_dim // 2, dtype=tf.float32)
        indices = tf.pow(10000.0, -2 * indices / self.output_dim)

        embeddings = tf.einsum("bn,d->bnd", position_ids, indices)
        embeddings = tf.stack([tf.sin(embeddings), tf.cos(embeddings)], axis=-1)
        embeddings = tf.reshape(embeddings, (-1, seq_len, self.output_dim))

        if self.merge_mode == "add":
            return inputs + embeddings
        if self.merge_mode == "concat":
            batch_size = tf.shape(inputs)[0]
            embeddings = tf.tile(embeddings, [batch_size, 1, 1])
            return tf.concat([inputs, embeddings], axis=-1)
        if self.merge_mode == "mul":
            return inputs * (embeddings + 1.0)
        if self.merge_mode == "zero":
            return embeddings

    def compute_output_shape(self, input_shape):
        if self.merge_mode in ("add", "mul", "zero"):
            return (input_shape[0], input_shape[1], self.output_dim)
        # concatenate
        return (input_shape[0], input_shape[1], input_shape[2] + self.output_dim)

    def get_config(self):
        base = super(SinusoidalPositionEmbedding, self).get_config()
        configs = {
            "output_dim": self.output_dim,
            "merge_mode": self.merge_mode
        }
        return dict(list(base.items()) + list(configs.items()))

class PositionEmbedding(tf.keras.layers.Layer):
    """经典的可训练位置Embedding，作为SinusoidalPositionEmbedding
    的替代。FaceBook论文Convolutional Sequence to Sequence Learning
    也提及到该Position Embedding。"""

    def __init__(
        self,
        input_dim,
        output_dim,
        merge_mode="add",
        embeddings_initializer="zeros",
        **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        # 支持的最大长度
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert merge_mode in ("add", "concat", "mul", "zero")
        self.merge_mode = merge_mode
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        # 支持mask往下传递
        self.supports_masking = True

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            trainable=True
        )

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        embeddings = self.embeddings[:seq_len]
        # 扩展到整个batch
        embeddings = tf.expand_dims(embeddings, axis=0)

        if self.merge_mode == "add":
            return inputs + embeddings
        if self.merge_mode == "concat":
            batch_size = tf.shape(inputs)[0]
            embeddings = tf.tile(embeddings, [batch_size, 1, 1])
            return tf.concat([inputs, embeddings], axis=-1)
        if self.merge_mode == "mul":
            return inputs * (embeddings + 1.0)
        if self.merge_mode == "zero":
            return embeddings

    def compute_output_shape(self, input_shape):
        if self.merge_mode in ("add", "mul", "zero"):
            return (input_shape[0], input_shape[1], self.output_dim)
        # concatenate
        return (input_shape[0], input_shape[1], input_shape[2] + self.output_dim)

    def get_config(self):
        base = super(PositionEmbedding, self).get_config()
        configs = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "merge_mode": self.merge_mode,
            "embeddings_initializer": initializers.serialize(self.embeddings_initializer)
        }
        return dict(list(base.items()) + list(configs.items()))

class SimplePositionEmbedding(tf.keras.layers.Layer):
    """可学习的位置Embedding，一种比PositionEmbedding
    更简单的实现，不过难扩展推广。非Transformer情况下使用，
    建议使用该实现。"""

    def __init__(self, maxlen, output_dim, embeddings_initializer="uniform", **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=maxlen,
            output_dim=output_dim,
            initializer=self.embeddings_initializer
        )

    def call(self, inputs):
        # maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = tf.expand_dims(positions, axis=0)
        return self.embedding(positions)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def plot(self):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return
        pe = tf.convert_to_tensor(self.embedding.embeddings)
        plt.imshow(pe)
        plt.show()

class RelativePositionEmbedding(tf.keras.layers.Layer):
    """来自Google的论文，经典的相对位置编码，即把原来对(i,j)的依赖变为对i-j的依赖。
    需要配合Relation-aware Self-Attention使用。本项目中，华为的NEZHA使用到该位置编码。
    可参考论文：https://arxiv.org/abs/1803.02155。可以参考解读文章：
    https://allenwind.github.io/blog/9582/
    """

    def __init__(self, input_dim, output_dim, embeddings_initializer="zeros", **kwargs):
        super(RelativePositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.max_position = (input_dim - 1) // 2
        # self.supports_masking = True

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs):
        q, v = inputs
        # 计算(i-j)
        q_indices = tf.range(0, tf.shape(q)[1], dtype=tf.int32)
        q_indices = tf.expand_dims(q_indices, axis=1)
        v_indices = tf.range(0, tf.shape(v)[1], dtype=tf.int32)
        v_indices = tf.expand_dims(v_indices, axis=0)
        pos_ids = v_indices - q_indices
        # 截断，便于适应任意的距离
        pos_ids = tf.clip_by_value(pos_ids, -self.max_position, self.max_position)
        pos_ids = pos_ids + self.max_position
        return tf.gather(self.embeddings, pos_ids)

    def compute_mask(self, inputs, mask):
        return mask[0]

    def compute_output_shape(self, input_shape):
        return (None, None, self.output_dim)

    def get_config(self):
        base = super(RelativePositionEmbedding, self).get_config()
        configs = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "embeddings_initializer": initializers.serialize(self.embeddings_initializer),
        }
        return dict(list(base.items()) + list(configs.items()))

class EmbeddingProjector(tf.keras.layers.Layer):
    """Embedding投影, ALBERT的Embedding需要"""

    def __init__(
        self,
        hidden_size,
        embedding_size,
        project_embeddings_with_bias=True,
        **kwargs):
        super(EmbeddingProjector, self).__init__(**kwargs)

    def build(self, input_shape):
        self.projector = self.add_weight(
            name="projector",
            shape=(self.embedding_size, self.hidden_size),
            initializer="uniform"
        )
        if self.project_embeddings_with_bias:
            self.projector_bias = self.add_weight(
                name="projector_bias",
                shape=(self.hidden_size,)
            )

    def call(self, inputs):
        x = tf.matmul(inputs, self.projector_layer)
        if self.project_embeddings_with_bias:
            x = x + self.projector_bias
        return x
