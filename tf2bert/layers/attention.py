import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import initializers
from tensorflow.keras import activations

class MultiHeadAttention(tf.keras.layers.Layer):
    """经典的点积缩放注意力（scaled dot product Attention）的多头实现，
    MultiHeadAttention，基本实现可参考Tensorflow的官方实现，见文档：
    https://tensorflow.google.cn/api_docs/python/tf/keras/layers/MultiHeadAttention?hl=en
    参考论文：https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        num_heads,
        head_size,
        key_size=None,
        out_dim=None,
        use_bias=True,
        attention_dropout_rate=0.0,
        use_attention_scale=True,
        use_residual_attention=False,
        return_attention_scores=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        if out_dim is None:
            out_dim = num_heads * head_size
        self.out_dim = out_dim
        if key_size is None:
            key_size = head_size
        self.key_size = key_size
        # Attention中softmax后的dropout
        self.attention_dropout_rate = attention_dropout_rate
        self.use_bias = use_bias
        # 是否对注意力进行缩放
        self.use_attention_scale = use_attention_scale
        # 是否使用残差连接
        self.use_residual_attention = use_residual_attention
        self.return_attention_scores = return_attention_scores
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.supports_masking = True

    def build(self, input_shape):
        self.qw_dense = Dense(
            units=self.key_size * self.num_heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )
        self.kw_dense = Dense(
            units=self.key_size * self.num_heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )
        self.vw_dense = Dense(
            units=self.head_size * self.num_heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )
        self.ow_dense = Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

    def call(self, inputs, mask=None, **kwargs):
        # bias是Attention矩阵的偏置项，与位置编码和attention mask相关
        q, k, v, *bias = inputs
        if mask is None:
            mask = [None] * 3
        # q_mask对输入的query序列的mask，以便输出的padding部分置0
        # v_mask对输入的value序列的mask，防止attention读取padding部分
        # query, key, value有三种使用场景：
        # 1. encoder-decoder attention，query来自前一个decoder的输出，key value来自encoder
        # 2. encoder self-attention，query, key, value均来自上一层encoder
        # 3. decoder masked self-attention，query, key, value均来自前一层decoder，不过带下三角矩阵mask
        q_mask, k_mask, v_mask = mask[:3]

        # 保证在不同的空间中进行投影，提高模型表达能力
        qw = self.qw_dense(q)
        kw = self.kw_dense(k)
        vw = self.vw_dense(v)
        # (batch_size, seq_len, num_heads, tensor_dim)
        qw = tf.reshape(qw, (-1, tf.shape(q)[1], self.num_heads, self.key_size))
        kw = tf.reshape(kw, (-1, tf.shape(k)[1], self.num_heads, self.key_size))
        vw = tf.reshape(vw, (-1, tf.shape(v)[1], self.num_heads, self.head_size))

        qkvw = [qw, kw, vw]
        mask = [q_mask, v_mask]

        # TODO把mask移动到这里处理

        attn, scores = self._compute_attention(qkvw, bias, mask, **kwargs)
        attn = tf.reshape(attn, (-1, tf.shape(attn)[1], self.num_heads * self.head_size))
        # (batch_size, seq_len, out_dim)
        attn = self.ow_dense(attn)
        if self.return_attention_scores:
            return attn, scores
        return attn

    def _compute_attention(self, qkvw, bias, mask, **kwargs):
        """softmax(Q*K.T)*V"""
        qw, kw, vw = qkvw
        q_mask, v_mask = mask

        # dot product Attention Q*K.T
        # 参考论文：https://arxiv.org/abs/1706.03762
        # S = Q * K.T
        a = tf.einsum("bjhd,bkhd->bhjk", qw, kw)

        # 处理Attention mask和位置编码
        with_attention_mask = kwargs.get("with_attention_mask")
        with_position_bias = kwargs.get("with_position_bias")
        if with_attention_mask:
            attention_mask = bias[0]
        if with_position_bias == "relative_position":
            # 相对位置编码参考论文：https://arxiv.org/pdf/1803.02155.pdf
            # 这里的实现就是论文上的方法展开来直接计算
            n = int(with_attention_mask)
            position_bias = bias[n]
            a = a + tf.einsum("bjhd,jkd->bhjk", qw, position_bias)

        # scaled dot product Attention
        # 最直观的理解，softmax存在饱和区，容易发生梯度消失，通过缩放避免落入饱和区
        # 可参考论文：https://arxiv.org/abs/2002.07028
        if self.use_attention_scale:
            a = a / tf.sqrt(float(self.key_size))

        # 如果是语言模型，叠加下三角Mask矩阵或unilm Mask矩阵
        if with_attention_mask:
            # (batch_size, num_heads, q_len, k_len)
            # (1, 1, q_len, k_len)
            a = self._add_attention_mask(a, attention_mask)

        # 计算padding的mask，消除对softmax的影响
        a = self._compute_sequence_mask(a, v_mask, -1e12)
        # 归一化，目前最常见的是softmax，一些优化可以在此展开
        A = tf.math.softmax(a, axis=-1)
        if self.attention_dropout_rate != 0:
            A = Dropout(self.attention_dropout_rate)(A)
        # 加权平均
        attn = tf.einsum("bhjk,bkhd->bjhd", A, vw)

        if with_position_bias == "relative_position":
            attn = attn + tf.einsum("bhjk,jkd->bjhd", A, position_bias)
        # (batch_size, seq_len, num_heads, head_size)
        return attn, a

    def _add_position_bias(self):
        """处理位置编码，不同Transformer模型有不同的相对位置编码的种类"""

    def _add_attention_mask(self, a, attention_mask):
        """attention矩阵的mask扩展处理，如语言模型中的下三角mask"""
        return a + attention_mask

    def compute_mask(self, inputs, mask=None):
        """计算mask，需要考虑attention scores"""
        if mask is not None:
            if self.return_attention_scores:
                return [mask[0], None]
            return mask[0]
        return mask

    def _compute_sequence_mask(self, x, mask, value):
        if mask is None:
            return x

        mask = tf.cast(mask, x.dtype)
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.expand_dims(mask, axis=1)
        return x * mask + value * (1 - mask)

    def compute_output_shape(self, input_shape):
        attention_shape = (input_shape[0][0], input_shape[0][1], self.out_dim)
        # (batch_size, num_heads, q_len, k_len)
        scores_shape = (input_shape[0][0], self.num_heads, input_shape[0][1], input_shape[1][1])
        if self.return_attention_scores:
            return [attention_shape, scores_shape]
        return attention_shape

    def get_config(self):
        base = super(MultiHeadAttention, self).get_config()
        configs = {
            "num_heads": self.num_heads,
            "head_size": self.head_size,
            "key_size": self.key_size,
            "out_dim": self.out_dim,
            "use_bias": self.use_bias,
            "attention_dropout_rate": self.attention_dropout_rate,
            "use_attention_scale": self.use_attention_scale,
            "use_residual_attention": self.use_residual_attention,
            "return_attention_scores": self.return_attention_scores,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer
        }
        return dict(list(base.items()) + list(configs.items()))

class Attention(MultiHeadAttention):
    """普通的Attention"""

    def __init__(self, out_dim, **kwargs):
        super(Attention, self).__init__(1, out_dim, **kwargs)
        self.out_dim = out_dim
