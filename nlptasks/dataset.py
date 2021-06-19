import re
import glob
import random
import itertools
import collections
import pandas as pd
import numpy as np
import tensorflow as tf
from tf2bert.text.labels import bmes2iobes
from tf2bert.text.labels import bio2iobes
from tf2bert.text.labels import iobes2bio

# 数据可以根据路径位置关键字眼Google搜索
# ==================================================================================== #
# classification

_THUCNews = "/home/zhiwen/workspace/dataset/THUCNews-title-label.txt"
def load_THUCNews_title_label(file=_THUCNews, nobrackets=True):
    categoricals = {'体育': 0, '娱乐': 1, '家居': 2, '彩票': 3, 
                    '房产': 4, '教育': 5, '时尚': 6, '时政': 7, 
                    '星座': 8, '游戏': 9, '社会': 10, '科技': 11, 
                    '股票': 12, '财经': 13}

    with open(file, encoding="utf-8") as fd:
        text = fd.read()
    lines = text.split("\n")[:-1]
    np.random.shuffle(lines)
    titles = []
    labels = []
    for line in lines[:]:
        title, label = line.split("\t")
        if not title:
            continue

        # 去掉括号内容
        if nobrackets:
            title = re.sub("\(.+?\)", lambda x:"", title)
        titles.append(title)
        labels.append(label)
    categoricals = list(set(labels))
    categoricals.sort()
    categoricals = {label:i for i, label in enumerate(categoricals)}
    clabels = [categoricals[i] for i in labels]
    return titles, clabels, categoricals

_THUContent = "/home/zhiwen/workspace/dataset/THUCTC/THUCNews/**/*.txt"
def load_THUCNews_content_label(file=_THUContent):    
    files = glob.glob(file)
    random.shuffle(files)

    # 获取所有标签
    labels = set()
    for file in files:
        label = file.rsplit("/", -2)[-2]
        labels.add(label)

    # label to id
    categoricals = {}
    for i, label in enumerate(labels):
        categoricals[label] = i

    def Xy_generator(files, with_label=True):
        for file in files:
            label = file.rsplit("/", -2)[-2]
            with open(file, encoding="utf-8") as fd:
                content = fd.read().strip()
            content = content.replace("\n", "").replace("\u3000", "")
            if with_label:
                yield content, categoricals[label]
            else:
                yield content

    return Xy_generator, files, categoricals

_HOTEL = "/home/zhiwen/workspace/dataset/classification/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv"
def load_hotel_comment(file=_HOTEL):
    with open(file, encoding="utf-8") as fd:
        lines = fd.readlines()[1:]
    random.shuffle(lines)
    X = []
    y = []
    for line in lines:
        label, commet = line.strip().split(",", 1)
        if not commet:
            continue
        X.append(commet)
        y.append(int(label))
    categoricals = {"负面":0, "正面":1}
    return X, y, categoricals

_w100k = "/home/zhiwen/workspace/dataset/classification/weibo_senti_100k/weibo_senti_100k.csv"
def load_weibo_senti_100k(file=_w100k, noe=True):
    df = pd.read_csv(file)
    X = df.review.to_list()
    y = df.label.to_list()
    # 去 emoji 表情，提升样本训练难度
    if noe:
        X = [re.sub("\[.+?\]", lambda x:"", s) for s in X]

    nX = []
    ny = []
    for sample, label in zip(X, y):
        if not sample:
            continue
        nX.append(sample)
        ny.append(label) 
    categoricals = {"负面":0, "正面":1}
    return X, y, categoricals

_MOODS = "/home/zhiwen/workspace/dataset/classification/simplifyweibo_4_moods.csv"
def load_simplifyweibo_4_moods(file=_MOODS):
    df = pd.read_csv(file)
    X = df.review.to_list()
    y = df.label.to_list()
    categoricals = {"喜悦":0, "愤怒":1, "厌恶":2, "低落":3}
    return X, y, categoricals

def load_imdb(file=None):
    """加载imdb英文数据"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
    word2id = tf.keras.datasets.imdb.get_word_index()
    id2word = {j:i for i,j in word2id.items()}
    X = []
    y = []
    for sample, label in zip(x_train, y_train):
        X.append(" ".join([id2word.get(i, " ") for i in sample]))
        y.append(label)

    for sample, label in zip(x_test, y_test):
        X.append(" ".join([id2word.get(i, " ") for i in sample]))
        y.append(label)
    categoricals = {"正类": 1, "负类": 0}
    return X, y, categoricals

# ==================================================================================== #
# matching

_LCQMC = "/home/zhiwen/workspace/dataset/LCQMC/totals.txt"
def load_lcqmc(file=_LCQMC):
    with open(file, encoding="utf-8") as fd:
        lines = fd.readlines()
    random.shuffle(lines)
    X1 = []
    X2 = []
    y = []
    for line in lines:
        x1, x2, label = line.strip().split("\t")
        if len(x1) * len(x2) == 0:
            continue
        X1.append(x1)
        X2.append(x2)
        y.append(int(label))
    categoricals = {"匹配":1, "不匹配":0}
    return X1, X2, y, categoricals

path = "/home/zhiwen/workspace/dataset/matching/afqmc_public/"
def load_afqmc(file="train.json", path=path):
    path = path + file
    data = pd.read_json(path, lines=True)
    X1 = data["sentence1"].values
    X2 = data["sentence2"].values
    y = data["label"].values
    categoricals = {"匹配":1, "不匹配":0}
    # 考虑输出端还原相似的信息
    return X1, X2, y, categoricals

_ATEC_CCKS = "/home/zhiwen/workspace/dataset/matching/ATEC_CCKS/totals.txt"
def load_ATEC_CCKS(file=_ATEC_CCKS):
    # ATEC + CCKS
    return load_lcqmc(file)

_ATEC = "/home/zhiwen/workspace/dataset/matching/ATEC/totals.txt"
def load_ATEC(file=_ATEC, shuffle=True):
    with open(file, encoding="utf-8") as fd:
        lines = fd.readlines()
    if shuffle:
        random.shuffle(lines)
    X1 = []
    X2 = []
    y = []
    for line in lines:
        _id, x1, x2, label = line.strip().split("\t")
        X1.append(x1)
        X2.append(x2)
        y.append(int(label))
    categoricals = {"匹配":1, "不匹配":0}
    return X1, X2, y, categoricals

_BQ = "/home/zhiwen/workspace/dataset/matching/BQ_corpus/totals.txt"
def load_bq_corpus(file=_BQ, shuffle=True):
    # http://icrc.hitsz.edu.cn/Article/show/175.html
    # https://www.aclweb.org/anthology/D18-1536.pdf
    with open(file, encoding="utf-8") as fd:
        lines = fd.readlines()
    if shuffle:
        random.shuffle(lines)
    X1 = []
    X2 = []
    y = []
    for line in lines:
        try:
            x1, x2, label = line.strip().split(",")
        except ValueError:
            # 跳过591个坏样本
            continue
        X1.append(x1)
        X2.append(x2)
        y.append(int(label))
    categoricals = {"匹配":1, "不匹配":0}
    return X1, X2, y, categoricals

_SNLI = "/home/zhiwen/workspace/dataset/SNLI_Corpus/snli_1.0_{}.csv"
def load_snli(file, shuffle=True):
    """纯英文数据集"""
    assert file in ("train", "dev", "test")
    file = _SNLI.format(file)
    with open(file, encoding="utf-8") as fd:
        lines = fd.readlines()[1:]
    if shuffle:
        random.shuffle(lines)
    categoricals = {"neutral":0, "contradiction":1, "entailment":2}
    X1 = []
    X2 = []
    y = []
    for line in lines:
        try:
            label, x1, x2 = line.strip().split(",")
        except ValueError:
            continue

        if label not in categoricals.keys():
            continue

        X1.append(x1)
        X2.append(x2)
        y.append(categoricals[label])
    return X1, X2, y, categoricals

# ==================================================================================== #
# NER

def load_file(file, sep=" ", shuffle=True, with_labels=False):
    # 返回逐位置标注形式
    with open(file, encoding="utf-8") as fp:
        text = fp.read()
    lines = text.split("\n\n")
    if shuffle:
        random.shuffle(lines)
    X = []
    y = []
    for line in lines:
        if not line:
            continue
        chars = []
        tags = []
        for item in line.split("\n"):
            char, label = item.split(sep)
            if label.startswith("M"):
                # M -> I
                label = "I" + label[1:]
            chars.append(char)
            tags.append(label)
        X.append("".join(chars))
        y.append(iobes2bio(tags))
        assert len(chars) == len(tags)
    if with_labels:
        labels = set(itertools.chain(*y))
        return X, y, sorted(labels)
    return X, y

def load_dh_msra(file="dataset/dh_msra.txt", shuffle=True, with_labels=False):
    # for evaluatoin
    return load_file(file, "\t", shuffle, with_labels)

PATH_CPD = "dataset/ner/china-people-daily-ner-corpus/example.{}"
def load_china_people_daily(file, shuffle=True, with_labels=False):
    file = PATH_CPD.format(file)
    return load_file(file, " ", shuffle, with_labels)

# china_people_daily 简称
load_cpd = load_china_people_daily

PATH_MSRA = "dataset/ner/msra/{}.ner"
def load_msra(file, shuffle=True, with_labels=False):
    file = PATH_MSRA.format(file)
    return load_file(file, " ", shuffle, with_labels)

PATH_WB = "dataset/ner/weibo/{}.all.bmes"
def load_weibo(file, shuffle=True, with_labels=False):
    file = PATH_WB.format(file)
    return load_file(file, " ", shuffle, with_labels)

PATH_ON = "dataset/ner/ontonote4/{}.char.bmes"
def load_ontonote4(file, shuffle=True, with_labels=False):
    file = PATH_ON.format(file)
    with open(file, "r") as fp:
        text = fp.read()
    lines = text.splitlines()
    X = []
    y = []
    for line in lines:
        sentence, tags = line.split("\t")
        X.append(sentence.replace(" ", ""))
        y.append(bmes2iobes(tags.split(" ")))
    if with_labels:
        labels = set(itertools.chain(*y))
        return X, y, sorted(labels)
    return X, y

PATH_RM = "dataset/ner/resume/{}.char.bmes"
def load_resume(file, shuffle=True, with_labels=False):
    file = PATH_RM.format(file)
    with open(file, "r") as fp:
        text = fp.read()
    lines = text.splitlines()
    X = []
    y = []
    for line in lines:
        sentence, tags = line.split("\t")
        X.append(sentence.replace(" ", ""))
        y.append(bmes2iobes(tags.split(" ")))

    if with_labels:
        labels = set(itertools.chain(*y))
        return X, y, sorted(labels)
    return X, y

# ==================================================================================== #
# CWS

_DICT = "dataset/dict.txt"
def load_freq_words(file=_DICT, proba=False, prefix=False):
    # 词频表
    words = {}
    total = 0
    with open(file, "r") as fp:
        lines = fp.read().splitlines()
    for line in lines:
        word, freq = line.split(" ")[:2]
        freq = int(freq)
        words[word] = freq
        total += freq
        # 前缀字典
        if prefix:
            for i in range(len(word)):
                sw = word[:i+1]
                if sw not in words:
                    words[sw] = 0
    if proba:
        words = {i:j/total for i,j in words.items()}
    return words, total

def load_sentences():
    # 测试分词效果的句子
    texts = []
    texts.append("守得云开见月明")
    texts.append("乒乓球拍卖完了")
    texts.append("无线电法国别研究")
    texts.append("广东省长假成绩单")
    texts.append("欢迎新老师生前来就餐")
    texts.append("上海浦东开发与建设同步")
    texts.append("独立自主和平等互利的原则")
    texts.append("黑天鹅和灰犀牛是两个突发性事件")
    texts.append("黄马与黑马是马，黄马与黑马不是白马，因此白马不是马。")
    texts.append("The quick brown fox jumps over the lazy dog.")
    texts.append("人的复杂的生理系统的特性注定了一件事情，就是从懂得某个道理到执行之间，是一个漫长的回路。")
    texts.append("除了导致大批旅客包括许多准备前往台北采访空难的新闻记者滞留在香港机场，直到下午2:17分日本亚细亚航空公司开出第一班离港到台北的班机才疏导了滞留在机场的旅客。")
    return texts

_HUMAH = "/home/zhiwen/workspace/dataset/human-history人类简史-从动物到上帝.txt"
def load_human_history(file=_HUMAH):
    # 加载长文本
    with open(file, "r") as fp:
        text = fp.read()
    return text

_PKU = "/home/zhiwen/workspace/dataset/icwb2-data/training/msr_training.utf8"
def load_icwb2_pku(file=_PKU):
    with open(file, "r") as fp:
        text = fp.read()
    sentences = text.splitlines()
    sentences = [re.split("\s+", sentence) for sentence in sentences]
    sentences = [[w for w in sentence if w] for sentence in sentences]
    return sentences

_CTB6 = "dataset/cws/ctb6/"
def load_ctb6_cws(path=_CTB6, file="train.txt"):
    if not file.endswith(".txt"):
        file += ".txt"
    file = path + file
    # 复用load_icwb2_pku的加载方法
    return load_icwb2_pku(file)
