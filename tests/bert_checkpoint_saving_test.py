import numpy as np
from tf2bert.models import build_transformer
from tf2bert.text.tokenizers import Tokenizer

config_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_config.json"
checkpoint_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"

model1 = build_transformer(
    model="bert+encoder", 
    config_path=config_path, 
    checkpoint_path=checkpoint_path,
    with_wrapper=True,
    with_mlm=True,
    verbose=False
)

file = "../temp/bert_model.ckpt"
model1.save_checkpoint(file)

model2 = build_transformer(
    model="bert+encoder", 
    config_path=config_path, 
    checkpoint_path=None, # 不加载原来的checkpoint
    with_wrapper=True,
    with_mlm=True,
    verbose=False
)

model2.model.load_weights(file)

token_dict_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/vocab.txt"
tokenizer = Tokenizer(token_dict_path, use_lower_case=True)

sentence = "峨眉山月半轮秋，影入平羌江水流。夜发清溪向三峡，思君不见下渝州。"
token_ids, segment_ids = tokenizer.encode(sentence)
token_ids = np.array([token_ids])
segment_ids = np.array([segment_ids])

probs1 = model1.model.predict([token_ids, segment_ids])[0]
probs2 = model2.model.predict([token_ids, segment_ids])[0]

assert np.array_equal(probs1, probs2)

print(probs1)
print(probs2)
print(tokenizer.decode(probs1[1:-1].argmax(axis=1)))
print(tokenizer.decode(probs2[1:-1].argmax(axis=1)))
print("checkpoint saving testing pass...")
