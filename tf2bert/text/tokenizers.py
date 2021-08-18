import re
import os
import json
import itertools
import unicodedata
import collections
import numpy as np

class CharTokenizer:
    """字ID切分器，用在NLU任务上，不支持在Transformer中使用"""

    def __init__(self, mintf=16):
        self.char2id = {}
        self.MASK = 0
        self.UNKNOW = 1
        self.mintf = mintf

    def fit(self, X):
        # 建立词ID映射表
        chars = collections.defaultdict(int)
        for c in itertools.chain(*X):
            chars[c] += 1
        # 过滤低频词
        chars = {i:j for i, j in chars.items() if j >= self.mintf}
        # 0:MASK
        # 1:UNK
        for i, c in enumerate(chars, start=2):
            self.char2id[c] = i

    def transform(self, X):
        # 转成ID序列
        ids = []
        for sentence in X:
            s = []
            for char in sentence:
                s.append(self.char2id.get(char, self.UNKNOW))
            ids.append(s)
        return ids

    def encode(self, text):
        return self.transform([text])[0]

    def __len__(self):
        return len(self.char2id) + 2

    def save(self, file):
        with open(file, "w") as fp:
            json.dump(
                self.char2id,
                fp,
                indent=4,
                ensure_ascii=False,
                sort_keys=True
            )

    def load(self, file):
        with open(file, "r") as fp:
            self.char2id = json.load(fp)

def save_vocab(dict_path, token_dict, encoding="utf-8"):
    """保存词典为文件"""
    with open(dict_path, "w", encoding=encoding) as fp:
        for k, v in sorted(token_dict.items(), key=lambda s: s[1]):
            fp.write(k + "\n")

def load_vocab(dict_path, encoding="utf-8"):
    """加载Transformer中的vocab.txt文件"""
    token_dict = {}
    if not os.path.exists(dict_path):
        return token_dict
    with open(dict_path, encoding=encoding) as fp:
        lines = fp.read().splitlines()
    for line in lines:
        token = line.split()
        token = token[0] if token else line.strip()
        token_dict[token] = len(token_dict)
    return token_dict

class Tokenizer:
    """Transformer的Tokenizer，兼容各类Transformer模型。
    可参看：https://github.com/google-research/bert.git"""

    def __init__(
        self,
        token_dict_path="",
        token_dict=None,
        use_lower_case=False, 
        word_maxlen=100,
        max_vocab_size=-1,
        use_simplify_vocab=False,
        use_decode_postprocess=True, # decode后进行postprocess操作
        tokenize_preprocess=None, # 在tokenize前对文本进行tokenize_preprocess操作
        tokenize_postprocess=None, # 在tokenize后对文本进行tokenize_postprocess操作
        token_mapping=None, # 在tokenize后，对token进行mapping
        with_token_start=True,
        with_token_end=True
    ):
        self._token_dict =  token_dict or self.load_vocab(token_dict_path)
        self._token_dict_inv = {v:k for k, v in self._token_dict.items()}
        self._vocab_size = len(self._token_dict)
        self._word_maxlen = word_maxlen
        self._max_vocab_size = max_vocab_size
        self._use_lower_case = use_lower_case
        self._use_simplify_vocab = use_simplify_vocab
        self._use_decode_postprocess = use_decode_postprocess
        self._tokenize_preprocess = tokenize_preprocess
        self._tokenize_postprocess = tokenize_postprocess
        self._token_mapping = token_mapping or {}
        self._token_mapping_inv = {v:k for k, v in self._token_mapping.items()}
        self._token_unk = "[UNK]"
        self._token_pad = "[PAD]"
        self._token_start = "[CLS]" if with_token_start else None
        self._token_end = "[SEP]" if with_token_end else None
        self._token_mask = "[MASK]"
        for token in ["_token_unk", "_token_pad", "_token_start", "_token_end", "_token_mask"]:
            _token_id = self._token_dict.get(getattr(self, token), None)
            token = token + "_id"
            setattr(self, token, _token_id)

    def fit(self, texts):
        """学习词表，预训练模型无需再使用该接口"""

    def simplify_token_dict(self, token_dict):
        """化简token dict"""
        pass

    def load_vocab(self, dict_path, encoding="utf-8"):
        """加载Transformer中的vocab.txt文件"""
        token_dict = {}
        if not os.path.exists(dict_path):
            return token_dict
        with open(dict_path, encoding=encoding) as fp:
            lines = fp.read().splitlines()
        for line in lines:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)
        return token_dict

    def save_vocab(self, path, encoding="utf-8"):
        """保存词汇表"""
        with open(path, "w", encoding=encoding) as fp:
            json.dump(
                self._token_dict,
                fp,
                indent=4,
                ensure_ascii=False,
                sort_keys=False
            )

    def show_special(self):
        for token, _id in self._token_dict.items():
            if not self._is_unused(token):
                continue
            print(token, "=", _id)
        print(self._token_unk, "=", self._token_unk_id)
        print(self._token_pad, "=", self._token_pad_id)
        print(self._token_start, "=", self._token_start_id)
        print(self._token_end, "=", self._token_end_id)
        print(self._token_mask, "=", self._token_mask_id)

    def tokenize(self, text, maxlen=None):
        tokens = []
        if self._token_start is not None:
            tokens.append(self._token_start)
        for token in self._tokenize(text):
            token = self._token_mapping.get(token) or token
            tokens.append(token)
        if self._token_end is not None:
            tokens.append(self._token_end)        

        if maxlen is not None:
            index = int(self._token_end is not None) + 1
            self.truncating_sequences(maxlen, -index, tokens)
        return tokens

    def batch_tokenize(self, texts, maxlen=None):
        """tokenize的批量操作"""
        return [self.tokenize(text, maxlen) for text in texts]

    def _tokenize(self, text, tokenize_preprocess=True):
        """tokenize的复用部分"""
        if self._use_lower_case:
            text = text.lower()
            text = unicodedata.normalize("NFD", text)
            text = "".join([ch for ch in text if unicodedata.category(ch) != "Mn"])

        if tokenize_preprocess and self._tokenize_preprocess is not None:
            tokens = []
            for token in self._tokenize_preprocess(text):
                if token in self._token_dict:
                    tokens.append(token)
                else:
                    tokens.extend(self._tokenize(token, False))
            return tokens

        spaced = ""
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += " " + ch + " "
            elif self._is_space(ch):
                spaced += " "
            elif ord(ch) == 0 or ord(ch) == 65533 or self._is_control(ch):
                continue
            else:
                spaced += ch

        tokens = []
        for word in spaced.strip().split():
            tokens.extend(self.subword_tokenize(word))
        return tokens

    def subword_tokenize(self, word):
        """word分成subword"""
        if len(word) > self._word_maxlen:
            return [word]

        start = 0
        end = 0
        tokens = []
        while start < len(word):
            end = len(word)
            while end > start:
                sub = word[start:end]
                if start > 0:
                    sub = self.stemize(sub)
                if sub in self._token_dict:
                    break
                end -= 1
            if start == end:
                return [word]
            else:
                tokens.append(sub)
                start = end
        return tokens

    def compute_mapping(self, text, tokens):
        """计算原始文本text与tokenize后获得tokens的映射关系，主要是考虑到token可能
        会对应text中多个连续的char。用于序列标注中的标注一一对应。
    
        例如：“3月20” tokenize 会得到 ['[CLS]', '3', '月', '20', '[SEP]']
        '20' 这个 token 对应原来的两个char，['2', '0']，返回的 token mapping 为
        [[], [0], [1], [2, 3], []]
        那么对于序列标注问题就需要特别处理了。
        """
        if self._use_lower_case:
            text = text.lower()

        char_mapping = []
        normalized_text = "" 
        for i, ch in enumerate(text, start=0):
            if self._use_lower_case:
                ch = unicodedata.normalize("NFD", ch)
                ch = "".join([c for c in ch if unicodedata.category(c) != "Mn"])
            ch = "".join([c for c in ch if not (ord(c) == 0 or ord(c) == 65533 or self._is_control(c))])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        offset = 0
        mapping = []
        text = normalized_text
        for token in tokens:
            if self._is_special(token):
                mapping.append([])
            else:
                token = self.stem(token)
                start = offset + text[offset:].index(token)
                end = start + len(token)
                mapping.append(char_mapping[start:end])
                offset = end
        return mapping

    def truncating_sequences(self, maxlen, index, *sequences):
        """截断总长度使得不超过指定长度maxlen"""
        sequences = [s for s in sequences if s]
        while True:
            lengths = [len(s) for s in sequences]
            if sum(lengths) > maxlen:
                i = np.argmax(lengths)
                sequences[i].pop(index)
            else:
                return sequences

    def encode(self, text1, text2=None, maxlen=None, mode="SEE", **kwargs):
        """text转成token id和segment id, text2用于句子对任务，如匹配任务"""
        if isinstance(text1, str):
            tokens1 = self.tokenize(text1)
        else:
            tokens1 = text1

        if isinstance(text2, str):
            # [CLS]xx[SEP]xx[SEP]
            if mode == "SEE":
                idx = 1 if self._token_start else 0
                tokens2 = self.tokenize(text2)[idx:]
            # [CLS]xx[SEP][CLS]xx[SEP]
            elif mode == "SESE":
                tokens2 = self.tokenize(text2)
        else:
            tokens2 = text2

        if maxlen is not None:
            index = int(self._token_end is not None) + 1
            self.truncating_sequences(maxlen, -index, tokens1, tokens2)

        token_ids = []
        segment_ids = []
        token_ids1 = self.tokens_to_ids(tokens1)
        segment_ids1 = [0] * len(token_ids1)
        token_ids.extend(token_ids1)
        segment_ids.extend(segment_ids1)

        if text2 is not None:
            token_ids2 = self.tokens_to_ids(tokens2)
            segment_ids2 = [1] * len(token_ids2)
            token_ids.extend(token_ids2)
            segment_ids.extend(segment_ids2)
        return token_ids, segment_ids

    def batch_encode(self, texts1, texts2=None, maxlen=None, mode="SEE"):
        """encode的batch操作"""
        if texts2 is None:
            texts2 = [None] * len(texts1)
        batch_token_ids = []
        batch_segment_ids = []
        for text1, text2 in zip(texts1, texts2):
            token_ids, segment_ids = self.encode(text1, text2, maxlen, mode)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
        return batch_token_ids, batch_segment_ids

    def decode(self, ids, tokens=None, **kwargs):
        """id序列转为可读文本"""
        tokens = self.ids_to_tokens(ids) or tokens
        tokens = [token for token in tokens if not self._is_special(token)]

        text = ""
        for i, token in enumerate(tokens):
            if self._is_stem(token):
                text += token[2:]
            elif len(token) == 1 and self._is_cjk_character(token):
                text += token
            elif len(token) == 1 and self._is_punctuation(token):
                text += token
                text += " "
            elif i > 0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += " "
                text += token
        if self._use_decode_postprocess:
            return self.decode_postprocess(text)
        return text

    def batch_decode(self, batch_ids, tokens=None):
        return [self.decode(ids) for ids in batch_ids]

    def tokens_to_ids(self, tokens):
        """token序列转换为对应的id序列"""
        return [self.token_to_id(token) for token in tokens]

    def ids_to_tokens(self, ids):
        """id序列转换为对应的token序列"""
        return [self.id_to_token(i) for i in ids]

    def token_to_id(self, token):
        """token转换为对应的id"""
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, i):
        """id转换为对应的token"""
        return self._token_dict_inv[i]

    def decode_postprocess(self, text):
        text = re.sub(" +", " ", text)
        text = re.sub("\' (re|m|s|t|ve|d|ll) ", "\'\\1 ", text)
        punctuation = self._cjk_punctuation() + "{[(<+-/="
        punctuation_regex = "|".join([re.escape(p) for p in punctuation])
        punctuation_regex = "({}) ".format(punctuation_regex)
        text = re.sub(punctuation_regex, "\\1", text)
        text = re.sub("(\d\.) (\d)", "\\1\\2", text)
        return text.strip()

    def stemize(self, sub):
        return "##" + sub

    def __len__(self):
        return self._vocab_size

    @property
    def vocab_size(self):
        """返回词汇表大小"""
        return self._vocab_size

    @property
    def vocabulary(self):
        """返回词汇表"""
        return self._token_dict

    @staticmethod
    def stem(token):
        """获取token的词干"""
        if token[:2] == "##":
            return token[2:]
        return token

    @staticmethod
    def _cjk_punctuation():
        """标点符号集
        py3:'＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
        """
        return "\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002"

    @staticmethod
    def _is_cjk_punctuation(ch):
        pass

    @staticmethod
    def _is_cjk_character(ch):
        """判断是否是CJK类字符（中英文）
        https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
               0x2A700 <= code <= 0x2B73F or \
               0x3400 <= code <= 0x4DBF or \
               0x20000 <= code <= 0x2A6DF or \
               0x2B820 <= code <= 0x2CEAF or \
               0x2B740 <= code <= 0x2B81F or \
               0xF900 <= code <= 0xFAFF or \
               0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_punctuation(ch):
        """判断是否是标点符号类字符（包括全/半角）
        py3:unicodedata.category("/") == "Po"
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
               58 <= code <= 64 or \
               91 <= code <= 96 or \
               123 <= code <= 126 or \
               unicodedata.category(ch).startswith("P")

    @staticmethod
    def _is_space(ch):
        """判断是否为空格类型字符"""
        return unicodedata.category(ch) == "Zs" or \
               ch == " " or ch == "\n" or \
               ch == "\r" or ch == "\t"

    @staticmethod
    def _is_stem(token):
        """判断是否是subword"""
        return token[:2] == "##"

    @staticmethod
    def _is_decimal(token):
        """判断是否是数字"""
        pass

    @staticmethod
    def _is_normal(token):
        """判断是否为普通字符"""
        return token[:2] != "##"

    @staticmethod
    def _is_control(ch):
        """判断是否是控制类字符"""
        return unicodedata.category(ch) in ("Cc", "Cf")

    @staticmethod
    def _is_special(ch):
        """判断是否是[xxx]类型的符号"""
        return bool(ch) and (ch[0] == "[") and (ch[-1] == "]")

    @staticmethod
    def _is_unused(ch):
        """判断是否是[unusedxx]类型"""
        return bool(ch) and ch.startswith("[unused")
