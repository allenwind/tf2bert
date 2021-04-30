import numpy as np
import tensorflow as tf
from tf2bert.text.tokenizers import Tokenizer
from tf2bert.models import build_transformer
from tf2bert.models import list_transformers

# 测试BERT模型持久化并加载

config_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_config.json"
checkpoint_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
token_dict_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/vocab.txt"

model = build_transformer(
    model="bert", 
    config_path=config_path, 
    checkpoint_path=checkpoint_path,
    with_mlm=False,
    verbose=False
)

text = "空山不见人，但闻人语响。返景入深林，复照青苔上。"
tokenizer = Tokenizer(token_dict_path, use_lower_case=True)

token_ids, segment_ids = tokenizer.encode(text)
token_ids = np.array(token_ids)
segment_ids = np.array(segment_ids)

feature1 = model.predict([token_ids, segment_ids])

path = "../temp/bert.model"
model.save(path)
del model

model = tf.keras.models.load_model(path)
feature2 = model.predict([token_ids, segment_ids])

assert np.array_equal(feature1, feature2)

models = list_transformers(with_lm=False)
models.remove("nezha") # TODO
for model in models:
    print("="*20 + model + "="*20)
    model = build_transformer(
        model=model, 
        config_path=config_path, 
        checkpoint_path=None,
        with_mlm=False,
        verbose=False
    )

    model.save(path)
    print("saved")
    del model
    model = tf.keras.models.load_model(path)
    print("loaded")
