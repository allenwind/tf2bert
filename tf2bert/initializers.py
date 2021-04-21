import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers

class SinusoidalInitializer(initializers.Initializer):

    def __call__(self, shape, dtype=None):
        vocab_size, depth = shape
        embeddings = np.zeros(shape)
        for pos in range(vocab_size):
            for i in range(depth // 2):
                theta = pos / np.power(10000, 2.0 * i / depth)
                embeddings[pos, 2 * i] = np.sin(theta)
                embeddings[pos, 2 * i + 1] = np.cos(theta)
        return tf.cast(embeddings, dtype)

class AlphaSinCosInitializer(initializers.Initializer):
    """SinusoidalInitializer的另外一种实现，提供一个alpha缩放因子"""

    def __init__(self, alpha=1.0):
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

class TransitionMatrixInitializer(initializers.Initializer):
    """状态矩阵的初始化"""

    def __init__(self, trans):
        self.trans = trans

    def __call__(self, shape, dtype=None):
        return tf.cast(self.trans, dtype)

    def get_config(self):
        return {"trans": self.trans}

class WordEmbeddingInitializer(initializers.Initializer):
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

class HybridInitializer(initializers.Initializer):
    """字词混合初始化，词的向量是其对应字的向量的均值，
    UNK使用所有字向量的均值"""

    def __init__(self, output_dim, char2vec, hybrid2id):
        input_dim = len(hybrid2id) + 2
        self.embeddings = np.zeros((input_dim, output_dim))
        vecs = list(char2vec.values())
        self.embeddings[1] = np.mean(vecs, axis=0)
        for hybrid, _id in hybrid2id.items():
            if self._is_word(hybrid):
                vec = 0.0
                for char in hybrid:
                    vec += char2vec.get(char, self.embeddings[1])
                vec = vec / len(hybrid)
                self.embeddings[_id] = vec
            else:
                vec = char2vec.get(hybrid, self.embeddings[1])
                self.embeddings[_id] = vec

    def __call__(self, shape, dtype=None):
        return tf.cast(self.embeddings, dtype)

    def _is_word(self, hybrid):
        return len(hybrid) > 1
