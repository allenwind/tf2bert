# tf2bert

本项目旨在拥抱Python3和Tensorflow2.x以及Transformer、PTMs。

本框架的目录结构类似于Tensorflow2.x，包括子packages：
- layers
- match
- math
- models
- text
- activations
- callbacks
- initializers
- losses
- metrics
- optimizers
- utils


保存成checkpoint形式





## Transformers对比



| 模型比较          | BERT | RoBERTa | ALBERT | NEZHA | GPT  | GPT2                                 | GPT2ML | LM   | UniLM |
| ----------------- | ---- | ------- | ------ | ----- | ---- | ------------------------------------ | ------ | ---- | ----- |
| 输入              |      |         |        |       |      |                                      |        |      |       |
| Embedding         |      |         |        |       |      | Embedding-Token + Embedding-Position |        |      |       |
| PositionEmbedding |      |         |        |       |      | 经典的可训练PositionEmbedding        |        |      |       |
| Attention         |      |         |        |       |      |                                      |        |      |       |
|                   |      |         |        |       |      |                                      |        |      |       |
| 预训练            |      |         |        |       |      |                                      |        |      |       |
|                   |      |         |        |       |      |                                      |        |      |       |



## 权重





BERT:

- brightmart版roberta: https://github.com/brightmart/roberta_zh
- ymcui版roberta: https://github.com/ymcui/Chinese-BERT-wwm

RoBERTa:

- 





ALBERT    NEZHA    GPT    GPT2    GPT2ML