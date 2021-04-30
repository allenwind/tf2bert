import numpy as np
from tf2bert.text.tokenizers import Tokenizer
from tf2bert.text.utils import load_cws_sentences
from tf2bert.models import build_transformer

# BERT特征提取测试

config_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_config.json"
checkpoint_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
token_dict_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/vocab.txt"

tokenizer = Tokenizer(token_dict_path, use_lower_case=True)
model = build_transformer(
    model="bert", 
    config_path=config_path, 
    checkpoint_path=checkpoint_path,
    verbose=False
)

for sentence in load_cws_sentences():
    token_ids, segment_ids = tokenizer.encode(sentence)
    token_ids = np.array([token_ids])
    segment_ids = np.array([segment_ids])
    features = model.predict([token_ids, segment_ids])
    print(sentence)
    print(features.shape)
    print(features)
