import tensorflow as tf
from tensorflow.keras.layers import *

import tensorflow as tf

class WordEmbeddingInitializer(tf.keras.initializers.Initializer):
    """基于训练好的词向量初始化"""

    def __init__(self, vocab, path, kind="word2vec"):
        # gensim.models.Word2Vec.load
        # gensim.models.KeyedVectors.load
        if kind == "word2vec":
            from gensim.models import Word2Vec as Model
        else:
            from gensim.models import KeyedVectors as Model
        model = Model.load(path)
        self.vocab = vocab
        self.word2id = {w:i for i, w in enumerate(model.wv.index2word)}
        self.word2vec = model.wv.syn0
        self.input_dim = len(self.vocab) + 2
        self.output_dim = self.word2vec.shape[-1]

    def __call__(self, shape=None, dtype=None):
        # 0 for MASK
        # 1 for UNK
        if shape is None:
            shape = self.shape
        embeddings = np.zeros(shape)
        for word, _id in self.vocab.items():
            w2v_id = self.word2id.get(word, "UNK")
            if w2v_id != "UNK":
                embeddings[_id] = self.word2vec[w2v_id]
            else:
                embeddings[1] = np.zeros(shape=(1, shape[-1]))
        return tf.cast(embeddings, dtype=dtype)

    @property
    def shape(self):
        return (self.input_dim, self.output_dim)

class SinCosInitializer(tf.keras.initializers.Initializer):

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, shape, dtype=None):
        # 使用 numpy 初始化位置向量矩阵
        # embeddings.shape = (1, input_dim, output_dim)
        _, input_dim, output_dim = shape
        pos = np.arange(input_dim)[:, np.newaxis]
        i = np.arange(output_dim)[np.newaxis, :]
        angles = pos / np.power(10000, 2 * i * self.alpha / output_dim)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        embeddings = tf.cast(angles[np.newaxis, ...], dtype)
        return embeddings

class SinCosPositionEmbedding(tf.keras.layers.Layer):
    """SinCos位置编码"""

    def __init__(self, input_dim=50, output_dim=512, alpha=1.0, trainable=False, **kwargs):
        super(SinCosPositionEmbedding, self).__init__(**kwargs)  
        self.input_dim = input_dim # seq_len
        self.output_dim = output_dim # seq_dim
        self.alpha = alpha # 缩放因子
        self.trainable = trainable

    def build(self, input_shape):
        # embeddings.shape = (1, input_dim, output_dim)
        self.embeddings = self.add_weight(
            name="SinCosPositionEmbedding",
            shape=(1, self.input_dim, self.output_dim),
            initializer=SinCosInitializer(self.alpha),
            trainable=self.trainable
        )

    def call(self, inputs):
        # 根据输入的序列长度返回相应的位置编码
        seq_len = tf.shape(inputs)[1]
        return self.embeddings[:, :seq_len, :]

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], self.output_dim)

class LearnablePositionEmbedding(tf.keras.layers.Layer):
    """可学习的位置编码"""

    def __init__(
        self,
        input_dim=50,
        output_dim=512,
        embeddings_initializer="zeros",
        **kwargs
    ):
        super(LearnablePositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = tf.keras.initializers.get(
            embeddings_initializer)

    def build(self, input_shape):
        # embeddings.shape = (1, input_dim, output_dim)
        self.embeddings = self.add_weight(
            name="LearnablePositionEmbedding", 
            shape=(1, self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return self.embeddings[:, :seq_len, :]

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], self.output_dim)

class PositionEmbedding(tf.keras.layers.Layer):
    """可学习的位置Embedding，一种更简单的实现"""

    def __init__(self, maxlen, output_dim, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.output_dim = output_dim
        self.embedding = tf.keras.layers.Embedding(
            input_dim=maxlen,
            output_dim=output_dim
        )

    def call(self, inputs):
        # maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = tf.expand_dims(positions, axis=0)
        return self.embedding(positions)

    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)

    def plot(self):
        import matplotlib.pyplot as plt
        import numpy as np
        pe = tf.convert_to_tensor(self.embedding.embeddings)
        plt.imshow(pe)
        plt.show()

if __name__ == "__main__":
    # for testing
    import numpy as np
    import matplotlib.pyplot as plt
    a = np.random.randn(1, 150, 510)
    embeddings = SinCosPositionEmbedding(150, alpha=0.3)
    plt.imshow(embeddings(a)[0])
    plt.show()



class MultiHeadAttention(Layer):
    
    def __init__(
        self,
        num_heads,
        size_per_head,
        attention_dropout=0.1
    ):
        self.num_heads = num_heads
        self.size_per_head = size_per_head
        self.units = num_heads * size_per_head
        self.attention_dropout = attention_dropout

    def build(self, input_shape):
        self.q_dense = Dense(
            self.units,
            kernel_initializer="glorot_uniform",
            name="query"
        )
        self.k_dense = Dense(
            self.units,
            kernel_initializer="glorot_uniform",
            name="key"
        )
        self.v_dense = Dense(
            self.units,
            kernel_initializer="glorot_uniform",
            name="value"
        )
        self.dropout = Dropout(self.attention_dropout)

    def compute_mask(self, inputs, mask=None):
        # mask向下传递
        return mask

    def call(self, inputs, mask=None, training=None):
        pass

    def _transpose(self, tensor, seq_len):
        pass

class FFN(Layer):
    pass
