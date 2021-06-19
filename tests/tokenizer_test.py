from tf2bert.text.tokenizers import Tokenizer
from tf2bert.text.utils import load_cws_sentences

dict_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/vocab.txt"
tokenizer = Tokenizer(dict_path)

tokenizer.show_special()
tokenizer.save_vocab("../temp/vocab.txt")

for text in load_cws_sentences():
    token_ids, segment_ids = tokenizer.encode(text)
    print(text)
    print(tokenizer.tokenize(text))
    print(token_ids)
    print(segment_ids)
    print(tokenizer.decode(token_ids))
    print(tokenizer._token_mask_id)

tokenizer = Tokenizer(
    dict_path,
    use_lower_case=True,
    with_token_start=False,
    with_token_end=False
)

tokenizer.show_special()

for text in load_cws_sentences():
    token_ids, segment_ids = tokenizer.encode(text)
    print(text)
    print(tokenizer.tokenize(text))
    print(token_ids)
    print(segment_ids)
    print(tokenizer.decode(token_ids))
    print(tokenizer._token_mask_id)

tokenizer = Tokenizer(
    dict_path,
    use_lower_case=True,
    with_token_start=True,
    with_token_end=True
)
text1 = "守得云开见月明"
text2 = "黑天鹅和灰犀牛是两个突发性事件"
text3 = "人的复杂的生理系统的特性注定了一件事情，就是从懂得某个道理到执行之间，是一个漫长的回路。"
token_ids, segment_ids = tokenizer.encode(text1, text2, mode="SEE")
print(token_ids)
token_ids, segment_ids = tokenizer.encode(text1, text2, mode="SESE")
print(token_ids)
token_ids, segment_ids = tokenizer.encode(text1, maxlen=20)
print(token_ids)
