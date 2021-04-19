from tf2bert.text.tokenizers import Tokenizer

dict_path = '/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/vocab.txt'
text = "人的复杂的生理系统的特性注定了一件事情，就是从懂得某个道理到执行之间，是一个漫长的回路。"

tokenizer = Tokenizer(dict_path, use_lower_case=True)

token_ids, segment_ids = tokenizer.encode(text)
print(text)
print(tokenizer.tokenize(text))
print(token_ids)
print(segment_ids)
print(tokenizer.decode(token_ids))
print(tokenizer._token_mask_id)
