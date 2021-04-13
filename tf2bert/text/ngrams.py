import re
import collections
import heapq
import json
from operator import itemgetter

class _lazyproperty:
    """延时计算"""

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value

def cut_ngrams(text, n):
    size = len(text)
    for i in range(size):
        for j in range(1, n+1):
            if i + j <= size:
                k = i + j
                yield text[i:k], (i, k)

class NgramsCounter:
    """统计文本序列ngrams"""

    _pattern = re.compile("[^0-9a-zA-Z\u4E00-\u9FD5#&\._%\-]+")

    def __init__(self, n):
        self.n = n
        # 可以使用Trie优化存储
        self._ngrams = collections.defaultdict(int)

    def fit(self, texts):
        for text in texts:
            for block in self.split_blocks(text):
                for ngram, _ in self.cut_ngrams(block):
                    self._ngrams[ngram] += 1

    def yield_ngrams(self, texts):
        for text in texts:
            for block in self.split_blocks(text):
                for ngram, _ in self.cut_ngrams(block):
                    yield ngram

    def split_blocks(self, text):
        for block in self._pattern.split(text):
            if block:
                yield block

    def cut_ngrams(self, block):
        size = len(block)
        for i in range(size):
            for j in range(1, self.n+1):
                if i + j <= size:
                    k = i + j
                    yield block[i:k], (i, k)

    def save(self, file):
        with open(file, "w") as fp:
            json.dump(
                self._ngrams,
                fp,
                indent=4,
                ensure_ascii=False,
                sort_keys=True
            )

    def load(self, file):
        with open(file, "r") as fp:
            self._ngrams = json.load(fp)

    @property
    def ngrams(self):
        return self._ngrams

    @_lazyproperty
    def vocab(self):
        # 字频率表
        return {c:v for c,v in self._ngrams.items() if len(c) == 1}

    @_lazyproperty
    def vocab_size(self):
        return len(self.vocab)

    @_lazyproperty
    def total_size(self):
        return sum(self.vocab.values())

    def most_common(self, n=None):
        # 支持传入负数，表示least common
        if n is None:
            return sorted(self._ngrams.items(), key=itemgetter(1), reverse=True)
        if n > 0:
            return heapq.nlargest(n, self._ngrams.items(), key=itemgetter(1))
        return self.least_common(-n)

    def least_common(self, n):
        # 最不常见
        return heapq.nsmallest(n, self._ngrams.items(), key=itemgetter(1))

if __name__ == "__main__":
    # for testing
    texts = [
        "除了导致大批旅客包括许多准备前往台北采访空难的新闻记者滞留在香港机场，直到下午2:17分日本亚细亚航空公司开出第一班离港到台北的班机才疏导了滞留在机场的旅客。",
        "人的复杂的生理系统的特性注定了一件事情，就是从懂得某个道理到执行之间，是一个漫长的回路。"
    ]

    n = 5
    c = NgramsCounter(n)
    c.fit(texts)
    c.save("ngrams.txt")
    print(c.ngrams)
    print()
    print(c.most_common(100))
    print(c.vocab_size)
    print(c.total_size)

    for ngram in cut_ngrams(texts[0], n):
        print(ngram)
