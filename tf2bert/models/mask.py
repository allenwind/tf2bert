import tensorflow as tf
from tensorflow.keras.layers import Lambda

# 在tf2bert/tests下有这两种掩码的numpy验证

class LMMaskMixIn:
    """计算下三角Mask，用于语言模型。这里使用MixIn的写法，
    通过继承即可扩展Transformer的mask，但单独使用没有意义。
    MaskedAttention也使用此mask。"""

    def compute_attention_mask(self, inputs=None):
        if self.attention_mask is None:
            def compute_mask(s):
                seq_len = tf.shape(s)[1]
                indices = tf.range(0, seq_len)
                # tf.expand_dims ~ [None]
                mask = indices[None, :] <= indices[:, None]
                mask = tf.cast(mask, tf.float32)
                return - (1 - mask[None, None]) * 1e12

            x = self.inputs[0] # 使用Embedding-Token计算
            self.attention_mask = self.build_layer(
                inputs=x,
                layer=Lambda,
                function=compute_mask,
                name="Attention-LM-Mask"
            )
        return self.attention_mask

class UniLMMaskMixIn:
    """计算Segment的下三角Mask, 用于Seq2Seq任务。对于输入部分，
    做双向Attention，而对于输出，做单向Attention。
    可参考论文：https://arxiv.org/abs/1905.03197"""

    def compute_attention_mask(self, inputs=None):
        if self.attention_mask is None:
            def compute_mask(s):
                indices = tf.cumsum(s, axis=1)
                # tf.expand_dims ~ [None]
                mask = indices[:, None, :] <= indices[:, :, None]
                mask = tf.cast(mask, tf.float32)
                return - (1 - mask[:, None]) * 1e12

            x = self.inputs[1] # 使用Embedding-Segment计算
            self.attention_mask = self.build_layer(
                inputs=x,
                layer=Lambda,
                function=compute_mask,
                name="Attention-UniLM-Mask"
            )
        return self.attention_mask

def install_attention_mask(TransformerModel, mask):
    assert mask in ("lm-mask", "unilm-mask")

    class LanguageModelTransformer(LMMaskMixIn, TransformerModel):

        def __init__(self, *args, **kwargs):
            super(LanguageModelTransformer, self).__init__(*args, **kwargs)
            self.with_mlm = self.with_mlm or True

    class UniLMTransformer(UniLMMaskMixIn, TransformerModel):

        def __init__(self, *args, **kwargs):
            super(UniLMTransformer, self).__init__(*args, **kwargs)
            self.with_mlm = self.with_mlm or True

    if mask == "lm-mask":
        return LanguageModelTransformer
    if mask == "unilm-mask":
        return UniLMTransformer
