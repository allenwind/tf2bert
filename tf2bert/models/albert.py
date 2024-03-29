import tensorflow as tf
from tensorflow.keras.layers import *

from tf2bert.layers import MultiHeadAttention
from tf2bert.layers import LayerNormalization
from tf2bert.layers import FeedForward
from .bert import BERT

class ALBERT(BERT):
    """ALBERT在模型架构上和BERT一致，只不过使用层参数共享起到正则化作用，不同
    之处是ALBERT对Embedding层进行矩阵低秩分解。ALBERT与BERT的一个显著不同之
    处是在预训练阶段，模型训练任务把NSP（Next Sentence Prediction）任务改为
    SOP（Sentence-Order Prediction）任务。前者还包括的不同之处：
    - 跨层权重共享
    - 句子顺序预测（SOP，相当于加大学习难度。原来NSP可以通过上下文与主题进行判断，
      而SOP只能根据上下文判断）
    - Embedding矩阵低秩分解（其实就是onehot隐射到相对较低维度的向量，然后再使用
      Dense升维到Attention输入的维度，这样打幅减少参数量）
    """

    def build_hidden_layer(self, inputs, index):
        """TransformerBlock基本结构，这里加上Dropout，
        Attention->Dropout->Add->LN->FNN->Dropout->Add->LN"""
        attention_name = "Transformer-MultiHeadSelfAttention"
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
            num_heads=self.num_attention_heads,
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
        x = self.build_layer(
            inputs=x,
            layer=LayerNormalization,
            hidden_initializer=self.initializer,
            name="{}-Norm".format(attention_name)
        )

        # position-wise feed-forward networks
        feed_forward_name = "Transformer-FeedForward"
        xi = x
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
            hidden_initializer=self.initializer,
            name="{}-Norm".format(feed_forward_name)
        )
        return x

    def variables_mapping(self):
        mapping = super(ALBERT, self).variables_mapping()
        albert_prefix = "bert/encoder/transformer/group_0/inner_group_0/"
        mapping.update({
            "Transformer-MultiHeadSelfAttention": [
                albert_prefix + "attention_1/self/query/kernel",
                albert_prefix + "attention_1/self/query/bias",
                albert_prefix + "attention_1/self/key/kernel",
                albert_prefix + "attention_1/self/key/bias",
                albert_prefix + "attention_1/self/value/kernel",
                albert_prefix + "attention_1/self/value/bias",
                albert_prefix + "attention_1/output/dense/kernel",
                albert_prefix + "attention_1/output/dense/bias"
            ],
            "Transformer-MultiHeadSelfAttention-Norm": [
                albert_prefix + "LayerNorm/beta",
                albert_prefix + "LayerNorm/gamma"
            ],
            "Transformer-FeedForward": [
                albert_prefix + "ffn_1/intermediate/dense/kernel",
                albert_prefix + "ffn_1/intermediate/dense/bias",
                albert_prefix + "ffn_1/intermediate/output/dense/kernel",
                albert_prefix + "ffn_1/intermediate/output/dense/bias"
            ],
            "Transformer-FeedForward-Norm": [
                albert_prefix + "LayerNorm_1/beta",
                albert_prefix + "LayerNorm_1/gamma"
            ]
        })
        return mapping

class UnsharedALBERT(BERT):
    """权重不共享，直接在BERT上加载"""

    def variables_mapping(self):
        mapping = super(UnsharedALBERT, self).variables_mapping()
        albert_prefix = "bert/encoder/transformer/group_0/inner_group_0/"
        for index in range(self.num_hidden_layers):
            mapping.update({
                "Transformer-{}-MultiHeadSelfAttention".format(index): [
                    albert_prefix + "attention_1/self/query/kernel",
                    albert_prefix + "attention_1/self/query/bias",
                    albert_prefix + "attention_1/self/key/kernel",
                    albert_prefix + "attention_1/self/key/bias",
                    albert_prefix + "attention_1/self/value/kernel",
                    albert_prefix + "attention_1/self/value/bias",
                    albert_prefix + "attention_1/output/dense/kernel",
                    albert_prefix + "attention_1/output/dense/bias",
                ],
                "Transformer-{}-MultiHeadSelfAttention-Norm".format(index): [
                    albert_prefix + "LayerNorm/beta",
                    albert_prefix + "LayerNorm/gamma",
                ],
                "Transformer-{}-FeedForward".format(index): [
                    albert_prefix + "ffn_1/intermediate/dense/kernel",
                    albert_prefix + "ffn_1/intermediate/dense/bias",
                    albert_prefix + "ffn_1/intermediate/output/dense/kernel",
                    albert_prefix + "ffn_1/intermediate/output/dense/bias",
                ],
                "Transformer-{}-FeedForward-Norm".format(index): [
                    albert_prefix + "LayerNorm_1/beta",
                    albert_prefix + "LayerNorm_1/gamma",
                ],
            })
        return mapping

