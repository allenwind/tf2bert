import tensorflow as tf
from tensorflow.keras.layers import *

from tf2bert.layers import PositionEmbedding
from tf2bert.layers import Embedding
from tf2bert.layers import LayerNormalization
from .gpt import GPT

class GPT2(GPT):
    """GPT2网络架构调整和特点：
    - LayerNormalization移动到每个子模块输入之后
    - 在最终的SelfAttention后添加LayerNormalization
    - Byte Pair Encoding

    参考：https://github.com/openai/gpt-2"""

    norm_epsilon = 1e-5

    def build_inputs(self):
        """GPT2的输入只有token ids序列, 去掉了segment ids"""
        x = self.build_layer(layer=Input, shape=(self.sequence_length,), name="Input-Token")
        inputs = [x]
        return inputs

    def build_embeddings(self, inputs):
        """GPT2的Embedding = Embedding-Token + Embedding-Position
        但二者叠加后不接LayerNormalization层。"""
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
        x = self.build_layer(
            inputs=x,
            layer=PositionEmbedding,
            input_dim=self.max_position,
            output_dim=self.embedding_size,
            merge_mode="add",
            embeddings_initializer=self.initializer,
            name="Embedding-Position"
        )
        # GPT2不接LayerNormalization
        if self.embedding_size != self.hidden_size:
            x = self.build_layer(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name="Embedding-Mapping"
            )
        return x

    def build_hidden_layers(self, inputs, index):
        """GPT2的隐层结构，
        LN->Attention->Dropout->Add->LN->FFN->Dropout->Add"""
        attention_name = "Transformer-{}-MultiHeadSelfAttention".format(index)
        callkwargs = {"with_attention_mask": True, "with_position_bias": False}
        x = inputs
        xi = x

        x = self.build_layer(
            inputs=x,
            layer=LayerNormalization,
            epsilon=self.norm_epsilon,
            hidden_initializer=self.initializer,
            name="{}-Norm".format(attention_name)
        )

        x = [x, x, x]
        attention_mask = self.compute_attention_bias(index)
        if attention_mask is not None:
            x.append(attention_mask)
            callkwargs["with_attention_mask"] = True

        x = self.build_layer(
            inputs=x,
            layer=MultiHeadAttention,
            callkwargs=callkwargs,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.build_layer(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name="{}-Dropout".format(attention_name)
        )
        x = self.build_layer(
            inputs=[xi, x],
            layer=Add,
            name="{}-Add".format(attention_name)
        )

        # position-wise feed-forward networks
        feed_forward_name = "Transformer-{}-FeedForward".format(index)
        xi = x
        x = self.build_layer(
            inputs=x,
            layer=LayerNormalization,
            epsilon=self.norm_epsilon,
            hidden_initializer=self.initializer,
            name="{}-Norm".format(feed_forward_name)
        )
        x = self.build_layer(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.build_layer(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name="{}-Dropout".format(feed_forward_name)
        )
        x = self.build_layer(
            inputs=[xi, x],
            layer=Add,
            name="{}-Add".format(feed_forward_name)
        )
        return x

    def build_outputs(self, inputs):
        x = inputs
        x = self.build_layer(
            inputs=x,
            layer=LayerNormalization,
            epsilon=self.norm_epsilon,
            hidden_initializer=self.initializer,
            name="Output-Norm"
        )
        x = self.build_layer(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name="Output-Dropout"
        )
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
        for k, v in super(GPT2, self).variables_mapping().items():
            mapping[k] = [i.replace("output/LayerNorm", "input/LayerNorm") for i in v]

        mapping["Output-Norm"] = [
            "gpt/output/LayerNorm/beta",
            "gpt/output/LayerNorm/gamma",
        ]
        return mapping
