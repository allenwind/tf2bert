import numpy as np
from tf2bert.text.tokenizers import Tokenizer

text = "3月20日下午2:17分日本亚细亚航空公司开出第一班离港到台北的班机1859才疏导了滞留在机场的旅客。"
dict_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/vocab.txt"

tokenizer = Tokenizer(
    dict_path,
    use_lower_case=True,
    with_token_start=True,
    with_token_end=True
)

tokens = tokenizer.tokenize(text, maxlen=512)
mapping = tokenizer.compute_mapping(text, tokens)
token_ids = tokenizer.tokens_to_ids(tokens)
print(tokens)
print(mapping)
print(token_ids)

# 根据mapping和text还原tokens
for token, j in zip(tokens[1:-1], mapping[1:-1]):
    if len(j) != 1:
        c = text[j[0]:j[-1]+1]
    else:
        c = text[j[0]]
    if token.startswith("##"):
        token = token[2:]
    print(token, c)
    assert token == c

text = "无3月20日无"
tag = "BBIIIIB"
tokens = tokenizer.tokenize(text, maxlen=512)
mapping = tokenizer.compute_mapping(text, tokens)
token_ids = tokenizer.tokens_to_ids(tokens)
print(tokens)
print(mapping)
print(token_ids)

# 序列标注
start_mapping = {j[0]:i for i,j in enumerate(mapping) if j}
end_mapping = {j[-1]:i for i,j in enumerate(mapping) if j}

start = text.index("3月20日")
end = start + len("3月20日")
print(text[start:end])
labels = np.zeros(len(token_ids))
if start in start_mapping and end in end_mapping:
    start = start_mapping[start]
    end = end_mapping[end]
    labels[start] = 1
    labels[start+1:end] = 2
print(list(text))
print(list(tag))
print(labels)
