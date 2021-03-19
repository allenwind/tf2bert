import re
import glob
import random
import itertools
import collections
import pandas as pd
import numpy as np

# ==================================================================================== #

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
    for line in lines:
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

# ==================================================================================== #

_LCQMC = "/home/zhiwen/workspace/dataset/LCQMC/totals.txt"
def load_lcqmc(file=_LCQMC):
    with open(file, encoding="utf-8") as fd:
        lines = fd.readlines()
    random.shuffle(lines)
    X1 = []
    X2 = []
    y = []
    for line in lines:
        if len(x1) * len(x2) == 0:
            continue
        x1, x2, label = line.strip().split("\t")
        X1.append(x1)
        X2.append(x2)
        y.append(int(label))
    categoricals = {"匹配":1, "不匹配":0}
    return X1, X2, y, categoricals
