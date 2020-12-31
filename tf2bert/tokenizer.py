
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

    def encode(self, ):
        pass





    def _is_space(ch):
        """空格类字符判断
        """

        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
            unicodedata.category(ch) == 'Zs'
