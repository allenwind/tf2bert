import random
import numpy as np
from tf2bert.text.tokenizers import Tokenizer
from tf2bert.text.utils import load_cws_sentences
from tf2bert.text.rendering import print_color_text
from tf2bert.models import build_transformer

# BERT掩码语言模型测试

config_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_config.json"
checkpoint_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
token_dict_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/vocab.txt"

tokenizer = Tokenizer(token_dict_path, use_lower_case=True)
model = build_transformer(
    model="bert", 
    config_path=config_path, 
    checkpoint_path=checkpoint_path,
    with_mlm=True,
    verbose=False
)

keywords = ["中国", "美国", "科技", "登月"]

sentence = "被鲨鱼攻击致死或被出故障的飞机碎片砸死，这两者中哪一种导致死亡的概率更大？幸运的是大部分人都没有经历过这两件事情，但如果问起这个问题，他们多半认为前者的概率更高。这个答案是错误的。在美国，被出故障的飞机碎片砸死的人数大概是被鲨鱼攻击而死亡的人数的3倍。"
import jieba

keywords = [w for w in jieba.lcut(sentence) if len(w) > 1]

token_ids = []
ws = []
for word in keywords:
    token_ids.extend(tokenizer.encode(word)[0][1:-1])
    ws.extend([0] * len(word))
    mask_nums = random.randint(1, 4)
    token_ids.extend([tokenizer._token_mask_id] * mask_nums)
    ws.extend([1] * mask_nums)
    
token_ids = [tokenizer._token_start_id] + token_ids + [tokenizer._token_end_id]
token_ids = np.array([token_ids])
segment_ids = np.array([0] * len(token_ids))
probs = model.predict([token_ids, segment_ids])[0]
# 红色部分为预测部分
print(keywords)
print(token_ids)
print_color_text(tokenizer.decode(probs[1:-1].argmax(axis=1)), ws)
print()

