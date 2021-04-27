import tensorflow as tf
from tensorflow.keras.layers import *

from tf2bert.layers import BiasAdd, PositionEmbedding
from tf2bert.layers import MultiHeadAttention
from tf2bert.layers import LayerNormalization
from tf2bert.layers import FeedForward, Embedding
from tf2bert.layers import DenseEmbedding
from .transformer import Transformer

class BERT(Transformer):

    def __init__(
        self,
        max_position,
        with_mlm=False,
        with_pool=False,
        with_nsp=False,
        segment_size=2,
        pool_activation="tanh",
        mlm_activation="softmax",
        pooler_type="first_token_transform",
        unshared_weights=True,
        **kwargs):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.with_mlm = with_mlm
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_pool = self.with_nsp or self.with_pool
        self.segment_size = segment_size
        self.pool_activation = pool_activation
        self.mlm_activation = mlm_activation
        # 指定pooler的类型，默认使用[CLS]
        self.pooler_type = pooler_type
        # 是否进行权重共享
        self.unshared_weights = unshared_weights

    def build_inputs(self):
        """InputToken + InputSegment"""
        x_in = self.build_layer(layer=Input, shape=(self.sequence_length,), name="Input-Token")
        s_in = self.build_layer(layer=Input, shape=(self.sequence_length,), name="Input-Segment")
        inputs = [x_in]
        if self.segment_size > 0:
            inputs.append(s_in)
        return inputs

    def build_embeddings(self, inputs):
        """BERT的Embedding = Embedding-Token + Embedding-Position + Embedding-Segment"""
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
            hidden_initializer=self.initializer,
            name="Embedding-Norm"
        )
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

    def build_hidden_layer(self, inputs, index):
        """TransformerBlock基本结构，Dropout是BERT抑制过拟合的基本方案
        Attention->Dropout->Add->LN->FNN->Dropout->Add->LN"""
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
            num_heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            attention_dropout_rate=self.attention_dropout_rate,
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
        feed_forward_name = "Transformer-{}-FeedForward".format(index)
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

    def build_outputs(self, inputs):
        x = inputs
        outputs = [x]

        if self.with_pool:
            x = outputs[0]
            x = self.build_layer(
                inputs=x,
                layer=Lambda,
                function=lambda x: x[:, 0],
                name="Pooler"
            )
            x = self.build_layer(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                activation=self.pool_activation,
                kernel_initializer=self.initializer,
                name="Pooler-Dense"
            )
            if self.with_nsp:
                x = self.build_layer(
                    inputs=x,
                    layer=Dense,
                    units=2,
                    activation="softmax",
                    kernel_initializer=self.initializer,
                    name="NSP-Proba"
                )
            outputs.append(x)

        if self.with_mlm:
            x = outputs[0]
            x = self.build_layer(
                inputs=x,
                layer=Dense,
                units=self.embedding_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name="MLM-Dense"
            )
            x = self.build_layer(
                inputs=x,
                layer=LayerNormalization,
                hidden_initializer=self.initializer,
                name="MLM-Norm"
            )
            x = self.build_layer(
                inputs=x,
                layer=Embedding,
                callkwargs={"mode": "dot"},
                name="Embedding-Token"
            )
            x = self.build_layer(inputs=x, layer=BiasAdd, name="MLM-Bias")
            x = self.build_layer(
                inputs=x,
                layer=Activation,
                activation=self.mlm_activation,
                name="MLM-Activation"
            )
            outputs.append(x)

        size = len(outputs)
        if size <= 2:
            outputs = outputs[size-1]
        else:
            outputs = outputs[1:]
        return outputs

    def _build_mlm_outputs(self, inputs):
        """MLM的输出"""
        x = inputs[0]
        x = self.build_layer(
            inputs=x,
            layer=Dense,
            units=self.embedding_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name="MLM-Dense"
        )
        x = self.build_layer(
            inputs=x,
            layer=LayerNormalization,
            hidden_initializer=self.initializer,
            name="MLM-Norm"
        )
        x = self.build_layer(
            inputs=x,
            layer=Embedding,
            callkwargs={"mode": "dot"},
            name="Embedding-Token"
        )
        x = self.build_layer(inputs=x, layer=BiasAdd, name="MLM-Bias")
        x = self.build_layer(
            inputs=x,
            layer=Activation,
            activation=self.mlm_activation,
            name="MLM-Activation"
        )
        return x

    def _build_pool_outputs(self, inputs):
        """Pooler的输出"""
        x = inputs[0]
        x = self.build_layer(
            inputs=x,
            layer=Lambda,
            function=lambda x: x[:, 0],
            name="Pooler"
        )
        x = self.build_layer(
            inputs=x,
            layer=Dense,
            units=self.hidden_size,
            activation=self.pool_activation,
            kernel_initializer=self.initializer,
            name="Pooler-Dense"
        )
        if self.with_nsp:
            x = self.build_layer(
                inputs=x,
                layer=Dense,
                units=2,
                activation="softmax",
                kernel_initializer=self.initializer,
                name="NSP-Proba"
            )
        return x

    def variables_mapping(self):
        """本地权重命名映射到BERT官方权重命名"""
        mapping = {
            "Embedding-Token": ["bert/embeddings/word_embeddings"],
            "Embedding-Segment": ["bert/embeddings/token_type_embeddings"],
            "Embedding-Position": ["bert/embeddings/position_embeddings"],
            "Embedding-Norm": [
                "bert/embeddings/LayerNorm/beta",
                "bert/embeddings/LayerNorm/gamma",
            ],
            "Embedding-Mapping": [
                "bert/encoder/embedding_hidden_mapping_in/kernel",
                "bert/encoder/embedding_hidden_mapping_in/bias",
            ],
            "Pooler-Dense": [
                "bert/pooler/dense/kernel",
                "bert/pooler/dense/bias",
            ],
            "NSP-Proba": [
                "cls/seq_relationship/output_weights",
                "cls/seq_relationship/output_bias",
            ],
            "MLM-Dense": [
                "cls/predictions/transform/dense/kernel",
                "cls/predictions/transform/dense/bias",
            ],
            "MLM-Norm": [
                "cls/predictions/transform/LayerNorm/beta",
                "cls/predictions/transform/LayerNorm/gamma",
            ],
            "MLM-Bias": ["cls/predictions/output_bias"],
        }

        for index in range(self.num_hidden_layers):
            bert_prefix = "bert/encoder/layer_{}/".format(index)
            mapping.update({
                "Transformer-{}-MultiHeadSelfAttention".format(index): [
                    bert_prefix + "attention/self/query/kernel",
                    bert_prefix + "attention/self/query/bias",
                    bert_prefix + "attention/self/key/kernel",
                    bert_prefix + "attention/self/key/bias",
                    bert_prefix + "attention/self/value/kernel",
                    bert_prefix + "attention/self/value/bias",
                    bert_prefix + "attention/output/dense/kernel",
                    bert_prefix + "attention/output/dense/bias",
                ],
                "Transformer-{}-MultiHeadSelfAttention-Norm".format(index): [
                    bert_prefix + "attention/output/LayerNorm/beta",
                    bert_prefix + "attention/output/LayerNorm/gamma",
                ],
                "Transformer-{}-FeedForward".format(index): [
                    bert_prefix + "intermediate/dense/kernel",
                    bert_prefix + "intermediate/dense/bias",
                    bert_prefix + "output/dense/kernel",
                    bert_prefix + "output/dense/bias",
                ],
                "Transformer-{}-FeedForward-Norm".format(index): [
                    bert_prefix + "output/LayerNorm/beta",
                    bert_prefix + "output/LayerNorm/gamma",
                ],
            })
        return mapping


