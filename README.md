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

