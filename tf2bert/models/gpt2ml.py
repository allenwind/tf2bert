import tensorflow as tf
from tensorflow.keras.layers import *

from tf2bert.layers import PositionEmbedding
from tf2bert.layers import Embedding
from tf2bert.layers import LayerNormalization
from .gpt import GPT

class GPT2ML(GPT):
    """GPT2ML:GPT2 for Multiple Languages
    - 多语言支持
    - 简化整理GPT2训练代码

    参看：https://github.com/imcaspar/gpt2-ml/blob/master/README_CN.md
    """

    norm_epsilon = 1e-5

    def build_inputs(self):
        """GPT2ML的输入只有token ids序列, 去掉了segment ids，和GPT2一致"""
        x = self.build_layer(layer=Input, shape=(self.sequence_length,), name="Input-Token")
        inputs = [x]
        return inputs

    def build_embeddings(self, inputs):
        """GPT2ML的Embedding = Embedding-Token + Embedding-Position
        二者叠加接LayerNormalization层。"""
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
        x = self.build_layer(
            inputs=x,
            layer=LayerNormalization,
            epsilon=self.norm_epsilon,
            hidden_initializer=self.initializer,
            name="Embedding-Norm"
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

    def build_hidden_layers(self, inputs, index):
        """GPT2ML基本结构，
        Attention->Dropout->Add->LN->FNN->Dropout->Add->LN
        """
        attention_name = "Transformer-{}-MultiHeadSelfAttention".format(index)
        callkwargs = {"with_attention_mask": False, "with_position_bias": False}
        x = inputs
        xi = x
        # MultiHeadSelfAttention q=k=v
        x = [x, x, x]

        attention_mask = self.compute_attention_mask(index)
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
            name="{}-Norm-0".format(feed_forward_name)
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
        x = self.build_layer(
            inputs=x,
            layer=LayerNormalization,
            epsilon=self.norm_epsilon,
            hidden_initializer=self.initializer,
            name="{}-Norm-1".format(feed_forward_name)
        )
        return x

    def variables_mapping(self):
        mapping = {
            "Embedding-Token": ["newslm/embeddings/word_embed"],
            "Embedding-Position": ["newslm/embeddings/pos_embed"],
            "Embedding-Norm": [
                "newslm/embeddings/LayerNorm_embed_norm/beta",
                "newslm/embeddings/LayerNorm_embed_norm/gamma",
            ],
        }

        for index in range(self.num_hidden_layers):
            gpt2ml_prefix = "newslm/layer{:02d}/".format(index)
            mapping.update({
                "Transformer-{}-MultiHeadSelfAttention".format(index): [
                    gpt2ml_prefix + "query_layer/kernel",
                    gpt2ml_prefix + "query_layer/bias",
                    gpt2ml_prefix + "key_layer/kernel",
                    gpt2ml_prefix + "key_layer/bias",
                    gpt2ml_prefix + "value_layer/kernel",
                    gpt2ml_prefix + "value_layer/bias",
                    gpt2ml_prefix + "context_projection_layer/kernel",
                    gpt2ml_prefix + "context_projection_layer/bias",
                ],
                "Transformer-{}-FeedForward-Norm-0".format(index): [
                    gpt2ml_prefix + "LayerNorm_mlp_ln0/beta",
                    gpt2ml_prefix + "LayerNorm_mlp_ln0/gamma",
                ],
                "Transformer-{}-FeedForward".format(index): [
                    gpt2ml_prefix + "intermediate/kernel",
                    gpt2ml_prefix + "intermediate/bias",
                    gpt2ml_prefix + "output/kernel",
                    gpt2ml_prefix + "output/bias",
                ],
                "Transformer-{}-FeedForward-Norm-1".format(index): [
                    gpt2ml_prefix + "LayerNorm_mlp_ln1/beta",
                    gpt2ml_prefix + "LayerNorm_mlp_ln1/gamma",
                ],
            })
        return mapping
