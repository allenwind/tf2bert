import tensorflow as tf
from tensorflow.keras.layers import *

from tf2bert.layers import PositionEmbedding, Embedding
from .bert import BERT
from .mask import LMMaskMixIn

class GPT(LMMaskMixIn, BERT):
    """GPT是语言模型，Embedding层和输出以及Attention Mask与BERT有差异。
    https://github.com/openai/finetune-transformer-lm"""

    def __init__(self, max_position, segment_size=2, final_activation="softmax", **kwargs):
        super(GPT, self).__init__(max_position, segment_size=segment_size, **kwargs)
        self.final_activation = final_activation

    def build_embeddings(self, inputs):
        """GPT的Embedding = Embedding-Token + Embedding-Position + Embedding-Segment
        但三者叠加后不接LayerNormalization层。"""
        inputs = inputs[:]
        if self.segment_size > 0:
            x, s = inputs
        else:
            x = inputs[0]

        x = self.build_layer(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name="Embedding-Token"
        )
        if self.segment_size > 0:
            s = self.build_layer(
                inputs=s,
                layer=Embedding,
                input_dim=self.segment_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name="Embedding-Segment"
            )
            x = self.build_layer(
                inputs=[x, s],
                layer=Add,
                name="Embedding-Token-Segment"
            )
        x = self.build_layer(
            layer=PositionEmbedding,
            input_dim=self.max_position,
            output_dim=self.embedding_size,
            merge_mode="add",
            embeddings_initializer=self.initializer,
            name="Embedding-Position"
        )
        # GPT不接LayerNormalization
        x = self.build_layer(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name="Embedding-Dropout"
        )
        if self.embedding_size != self.hidden_size:
            x = self.build_layer(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name="Embedding-Mapping"
            )
        return x

    def build_outputs(self, inputs):
        """语言模型的输出"""
        x = inputs
        x = self.build_layer(
            inputs=x,
            layer=Embedding,
            callkwargs={"mode": "dot"},
            name="Embedding-Token"
        )
        x = self.build_layer(
            inputs=x,
            layer=Activation,
            activation=self.final_activation,
            name="LM-Activation"
        )
        return x

    def variables_mapping(self):
        mapping = {}
        for k, v in super(GPT, self).variables_mapping().items():
            mapping[k] = [i.replace("bert/", "gpt/").replace("encoder", "transformer") \
                          for i in v]
        return mapping
