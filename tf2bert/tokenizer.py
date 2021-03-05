
class Tokenizer:

    def __init__(
        self,
        tokens,
        start="[CLS]",
        end="[SEP]",
        tokenize=None,
        mapping=None
    ):
        pass
        self.spaces = set([" ", "\n", "\r", "\t"])

    def encode(self, *texts, maxlen):
        """输出token ids和segment ids"""
        pass

    def decode(self, ids):
        """把token ids转换为文本"""
        pass

    def tokenize(self, text):
        pass


    def _tokenize(self, text, ):
        pass


    def _is_space(ch):
        """空格类字符判断
        """

        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
            unicodedata.category(ch) == 'Zs'
