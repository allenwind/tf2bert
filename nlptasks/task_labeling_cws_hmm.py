import itertools
import random
import re
import math
import numpy as np
from collections import *
from tf2bert.text.labels import find_entities
from tf2bert.text.labels import find_words

import dataset

class HiddenMarkovChain:
    
    def __init__(self, tags, task="NER"):
        # 标签集
        self.tags = sorted(set(tags))
        self.tags2id = {i:j for j,i in enumerate(self.tags)}
        self.id2tags = {j:i for i,j in self.tags2id.items()}
        self.state_size = len(tags)
        assert task in ("NER", "CWS")
        if task == "NER":
            self.convert = find_entities
        else:
            self.convert = find_words
        self.reset()

    def reset(self):
        # 初始状态的参数学习
        self.pi = np.zeros((1, self.state_size))
        # 状态转移矩阵
        self.A = np.zeros((self.state_size, self.state_size))
        # 观察矩阵，稀疏形式
        self.B = defaultdict(Counter)
        self.built = False

    def fit(self, X, y):
        if self.built:
            self.reset()
        # 状态转移矩阵参数学习
        for labels in y:
            for label1, label2 in zip(labels[:-1], labels[1:]):
                id1 = self.tags2id[label1]
                id2 = self.tags2id[label2]
                self.A[id1][id2] += 1
        self.A = self.A / np.sum(self.A, axis=1, keepdims=True)
        # 观察矩阵参数学习
        for sentence, labels in zip(X, y):
            for char, label in zip(sentence, labels):
                self.B[label][char] += 1
        self.logtotal = {tag:math.log(sum(self.B[tag].values())) for tag in self.tags}
        self.built = True

    def predict(self, X):
        # 给定一个batch的观察序列X，预测各个样本每个时间步隐状态的分值scores
        batch_scores = []
        for sentence in X:
            scores = np.zeros((len(sentence), self.state_size))
            for i, char in enumerate(sentence):
                for j, k in self.B.items():
                    scores[i][self.tags2id[j]] = math.log(k[char]+1) - self.logtotal[j]
            batch_scores.append(scores)
        return batch_scores

    def sampling(self, steps):
        init = self.pi
        rs = np.zeros((steps+1, self.state_size))

    def _sampling_from_multi_category(self, p):
        return np.random.multinomial(1, p)

    def find(self, sentence):
        # 用viterbi求scores最优路径
        scores = self.predict([sentence])[0]
        log_trans = np.log(np.where(self.A==0, 0.0001, self.A))
        viterbi = self.viterbi_decode(scores, log_trans)
        viterbi = [self.id2tags[i] for i in viterbi]
        return self.convert(sentence, viterbi)

    def viterbi_decode(self, scores, trans, return_score=False):
        # 使用viterbi算法求最优路径
        # scores.shape = (seq_len, num_tags)
        # trans.shape = (num_tags, num_tags)
        dp = np.zeros_like(scores)
        backpointers = np.zeros_like(scores, dtype=np.int32)
        dp[0] = scores[0]
        for t in range(1, scores.shape[0]):
            # 扩展维度便于广播，计算上一时间步到当前时间步所有路径分值
            v = np.expand_dims(dp[t-1], axis=1) + trans
            # 保存当前时间步各状态的最优路径
            dp[t] = scores[t] + np.max(v, axis=0)
            backpointers[t] = np.argmax(v, axis=0)

        # 回溯状态
        viterbi = [np.argmax(dp[-1])]
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        if return_score:
            viterbi_score = np.max(dp[-1])
            return viterbi, viterbi_score
        return viterbi

    def plot_trans(self):
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ImportError as err:
            print(err)
            return
        ax = sns.heatmap(
            self.A,
            vmin=0.0,
            vmax=1.0,
            fmt=".2f",
            cmap="copper",
            annot=True,
            cbar=True,
            # xticklabels=self.tags,
            # yticklabels=self.tags,
            linewidths=0.25,
            cbar_kws={"orientation": "horizontal"}
        )
        ax.set_title("Transition Matrix")
        ax.set_xticklabels(self.tags, rotation=0)
        ax.set_yticklabels(self.tags, rotation=0)
        plt.show()

class TokenizerBase:
    """分词的基类，继承该类并在find_word实现分词的核心算法"""

    spaces = re.compile("(\r\n|\s)", re.U)
    english = re.compile("[a-zA-Z0-9]", re.U)
    chinese = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)", re.U)

    def cut(self, text):
        return list(self._cut(text))

    def _cut(self, text):
        # 把长文本切分为句子块
        for block in self.chinese.split(text):
            if not block:
                continue
            if self.chinese.match(block):
                yield from self.cut_block(block)
            else:
                for s in self.spaces.split(block):
                    if self.spaces.match(s):
                        yield s
                    else:
                        yield from s

    def cut_block(self, sentence):
        # 对文本进行分块分句后分词
        buf = ""
        for word in self.find_word(sentence):
            if len(word) == 1 and self.english.match(word):
                buf += word
            else:
                if buf:
                    yield buf
                    buf = ""
                yield word
        if buf:
            yield buf

    def find_word(self, sentence):
        # 从这里实现分词算法的核心
        # 从句子中发现可以构成的词，返回可迭代对象
        raise NotImplementedError

class HMMTokenizer(TokenizerBase, HiddenMarkovChain):
    """分块后的HMM分词"""

    def find_word(self, sentence):
        yield from self.find(sentence)

def compute_sbme_tags(sentence):
    tags = []
    for word in sentence:
        if len(word) == 1:
            tags.append("S")
        else:
            tags.extend(["B"] + ["M"]*(len(word)-2) + ["E"])
    return np.array(tags)

def gen_random_sentences(nums, maxsize=100):
    try:
        import jieba
    except ImportError as err:
        print(err)
        return ""

    jieba.initialize()
    words = {w:v for w,v in jieba.dt.FREQ.items() if v != 0}
    ws = list(words.keys())
    p = np.array(list(words.values()))
    p = p / np.sum(p)
    for _ in range(nums):
        size = random.randint(10, maxsize)
        sentence = np.random.choice(ws, size, p=p)
        tags = compute_sbme_tags(sentence)
        yield "".join(sentence), tags

def load_random_sentences(file=None, nums=50000, with_labels=False):
    X = []
    y = []
    for text, tags in gen_random_sentences(nums):
        X.append(text)
        y.append(tags)
    if with_labels:
        labels = sorted("BMES")
        return X, y, labels
    return X, y

if __name__ == "__main__":
    # 构造随机句子样本
    X, y, labels = load_random_sentences("train", nums=1000, with_labels=True)

    # 标准HMM参数学习
    model = HiddenMarkovChain(labels, task="CWS")
    model.fit(X, y)
    model.plot_trans()

    # 分块后的HMM参数学习
    tokenizer = HMMTokenizer(labels, task="CWS")
    tokenizer.fit(X, y)

    # 两种方案对比
    for text in dataset.load_sentences():
        print(model.find(text))
        print(tokenizer.cut(text))

    sentences = dataset.load_ctb6_cws(file="test.txt")
    for sentence in sentences:
        labels = compute_sbme_tags(sentence)
        sentence = "".join(sentence)
        print(find_words(sentence, labels))
        print(model.find(sentence))
        print(tokenizer.cut(sentence))
        input()
