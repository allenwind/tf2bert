from tf2bert.text.tokenizers import Tokenizer
from tf2bert.text.utils import load_cws_sentences

dict_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/vocab.txt"
tokenizer = Tokenizer(dict_path)

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
