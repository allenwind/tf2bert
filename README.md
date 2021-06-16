# tf2bert

本项目旨在拥抱Python3和Tensorflow2.x以及Attention、Transformer、PTMs。

本框架的目录结构类似于Tensorflow2.x，包括子packages：
- layers
- math
- models
- text
- activations
- callbacks
- initializers
- losses
- metrics
- optimizers
- preprocessing
- utils

本项目只依赖`tensorflow2.x`、`tensorflow-addons`。目前支持BERT、RoBERTa、ALBERT、NEZHA、GPT等模型，包括CRF层（依赖`tensorflow-addons`）、Normalization、多种支持Mask的Pooling、Embedding（包括多种位置Embedding、HybridEmbedding）、常用激活函数、Metrics等等。



## 模型简述

简单梳理一下常见PTMs的特点：

| 模型    | 特点                                                         |
| ------- | ------------------------------------------------------------ |
| BERT    | 多层的Transformer Encoder堆叠而成、经典的可训练PositionEmbedding、MLM + NSP |
| ALBERT  | Factorized Embedding Parameterization、跨层共享参数、引入句子顺序预测（SOP） |
| RoBERTa | 中文WWM（Whole Word Masking）策略、动态mask、Tokenizer采用Byte Pair Encoding、去掉NSP引入SOP、MLM + SOP |
| ERNIE   | mask策略引入短语级别（phrase-level mask）与实体级别（entity-level mask）进而在模型中引入实体方面的先验知识 |
| NEZHA   | 改用经典的相对位置PositionEmbedding、优化算法[LAMB](https://arxiv.org/abs/1904.00962)加速训练 |
| GPT     | Transformer Decoder堆叠而成、语言模型、Embedding层叠加后不加LN |
| GPT2    | 更多参数更大的网络容量、LN移动到每个子模块输入之后、Attention后添加LN、输入去掉segment |
| GPT2ML  | 多语言支持、简化整理GPT2训练                                 |
| LM      | 计算下三角Mask，用于语言模型                                 |
| UniLM   | 通过Segment的下三角Mask，使得BERT支持Seq2Seq任务。Mask原理是，对于输入部分，做双向Attention，而对于输出，做单向Attention |



## 使用

本项目依赖：`tensorflow2.x`、`tensorflow-addons`。由于更新较快，不使用pip，推荐使用`PYTHONPATH`。

克隆项目到`{your_path}`，

```bash
git clone https://github.com/allenwind/tf2bert.git
```

当需要更新时，直接`git pull`获取源码更新。

打开`.bashrc`添加项目路径到`PYTHONPATH`环境变量中，

```.bashrc
export PYTHONPATH={your_path}/tf2bert:$PYTHONPATH
```

然后，

```bash
source ~/.bashrc
```

简单例子，

```python
import numpy as np
from tf2bert.text.tokenizers import Tokenizer
from tf2bert.text.utils import load_sentences
from tf2bert.models import build_transformer

config_path = "bert/chinese_L-12_H-768_A-12/bert_config.json"
checkpoint_path = "bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
token_dict_path = "bert/chinese_L-12_H-768_A-12/vocab.txt"

tokenizer = Tokenizer(token_dict_path)
model = build_transformer(
    model="bert+encoder", 
    config_path=config_path, 
    checkpoint_path=checkpoint_path,
    verbose=True
)

for sentence in load_sentences():
    token_ids, segment_ids = tokenizer.encode(sentence)
    token_ids = np.array([token_ids])
    segment_ids = np.array([segment_ids])
    features = model.predict([token_ids, segment_ids])
    print(sentence)
    print(features.shape)
    print(features)
```

更多的例子可参看`nlptasks`目录、`tests`目录代码。



## 相关链接

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)

[NEZHA: Neural Contextualized Representation for Chinese Language Understanding](https://arxiv.org/abs/1909.00204)

[Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)

[Are Pre-trained Convolutions Better than Pre-trained Transformers?](https://arxiv.org/abs/2105.03322)

[Synthesizer: Rethinking Self-Attention in Transformer Models](https://arxiv.org/abs/2005.00743)

[MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)



