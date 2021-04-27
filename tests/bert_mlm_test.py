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

rate = 0.3

for sentence in load_cws_sentences():
    token_ids, segment_ids = tokenizer.encode(sentence)
    ws = np.zeros((len(sentence),))

    mask_nums = int(rate * len(sentence))
    masks = random.sample(range(1, len(sentence)), mask_nums)
    for mask in masks:
        ws[mask] = 1
        token_ids[mask] = tokenizer._token_mask_id

    token_ids = np.array([token_ids])
    segment_ids = np.array([segment_ids])
    probs = model.predict([token_ids, segment_ids])[0]
    # 红色部分为预测部分
    print_color_text(sentence, ws)
    print_color_text(tokenizer.decode(probs[1:-1].argmax(axis=1)), ws)
    print()

"""
黑天鹅和灰犀牛是两个突发性事件
。天鹅和灰天鹅是两个突发性事件

黄马与黑马是马，黄马与黑马不是白马，因此白马不是马。
白马与白马是马，白马与黑马不是白马，因此白马不是马。

空山不见人，但闻人语响。返景入深林，复照青苔上。
青山不见人，但闻人不声。一景入深处，复照青苔上。

峨眉山月半轮秋，影入平羌江水流。夜发清溪向三峡，思君不见下渝州。
峨眉山月一轮秋，西见平羌山水寒。夜发清溪向三山，思君不见下荆州。

投资界是杂乱、艰难的世界，与我们十年前所熟悉的世界大不相同。我们将了解一系列特别重要的新的威胁、你如何处理它们，以及新的机遇。
投资料是杂乱、艰难的世界，与我们十年前所熟悉的世界大不相同。我们要了解一系列特别重要的新兴威胁、该如何处理它们，以及新的机遇。

人的复杂的生理系统的特性注定了一件事情，就是从懂得某个道理到执行之间，是一个漫长的回路。
。的复杂性和和系统的特性注定了一件事情，我们从懂得某个道理到一段时间，是一个漫长的回路。
"""
