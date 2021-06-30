import time
import numpy as np

def find_best_maxlen(X, mode="max"):
    # 获取适合的截断长度
    ls = [len(sample) for sample in X]
    if mode == "mode":
        maxlen = np.argmax(np.bincount(ls))
    if mode == "mean":
        maxlen = np.mean(ls)
    if mode == "median":
        maxlen = np.median(ls)
    if mode == "max":
        maxlen = np.max(ls)
    return int(maxlen)

def textQ2B(text):
    """把文本从全角符号转半角符号"""
    rtext = []
    for c in text:
        c = ord(c)
        # \u3000
        if c == 12288:
            c = 32
        # 全角字符
        elif (c >= 65281 and c <= 65374):
            c -= 65248
        rtext.append(chr(c))
    return "".join(rtext)

def humanize_bytes(bytesize, precision=3):
    abbrevs = (
        (1 << 50, "PB"),
        (1 << 40, "TB"),
        (1 << 30, "GB"),
        (1 << 20, "MB"),
        (1 << 10, "kB"),
        (1, "bytes")
    )
    if bytesize == 1:
        return "1 byte"
    for factor, suffix in abbrevs:
        if bytesize >= factor:
            break
    return "%.*f %s" % (precision, bytesize / factor, suffix)

def humanize_time(seconds):
    monthname = [None,
                 "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    year, month, day, hh, mm, ss, *_ = time.localtime(seconds)
    htime = "%02d/%3s/%04d %02d:%02d:%02d" % (day, monthname[month], year, hh, mm, ss)
    return htime

print(humanize_time(389084798))


class BatchContainer:
    """装载批量数据的容器"""

    def __init__(self, cell_nums):
        self.cell_nums = cell_nums
        self.init_cells()

    def init_cells(self):
        self.cells = []
        for _ in range(self.cell_nums):
            self.cells.append([])

    def append(self, *samples):
        for cell, sample in zip(self.cells, samples):
            cell.append(sample)

    def batch_pop(self):
        cells = self.cells
        self.init_cells()
        return cells

    @property
    def size(self):
        return len(self.cells[0]) if self.cells[0] else 0

def load_cws_sentences():
    """测试分词效果的句子"""
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
    texts.append("空山不见人，但闻人语响。返景入深林，复照青苔上。")
    texts.append("峨眉山月半轮秋，影入平羌江水流。夜发清溪向三峡，思君不见下渝州。")
    texts.append("投资界是杂乱、艰难的世界，与我们十年前所熟悉的世界大不相同。我们将了解一系列特别重要的新的威胁、你如何处理它们，以及新的机遇。")
    # texts.append("The quick brown fox jumps over the lazy dog.")
    texts.append("人的复杂的生理系统的特性注定了一件事情，就是从懂得某个道理到执行之间，是一个漫长的回路。")
    texts.append("被鲨鱼攻击致死或被出故障的飞机碎片砸死，这两者中哪一种导致死亡的概率更大？幸运的是大部分人都没有经历过这两件事情，但如果问起这个问题，他们多半认为前者的概率更高。这个答案是错误的。在美国，被出故障的飞机碎片砸死的人数大概是被鲨鱼攻击而死亡的人数的3倍。")
    texts.append("除了导致大批旅客包括许多准备前往台北采访空难的新闻记者滞留在香港机场，直到下午2:17分日本亚细亚航空公司开出第一班离港到台北的班机才疏导了滞留在机场的旅客。")
    return texts

def load_ner_sentences():
    """NER句子示例"""
    texts = []
    labels = []
    texts.append("志愿者们和市里、地区乃至广西团区委、团中央的干部都成了好朋友。")
    labels.append(["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"])
    texts.append("目前申花队、雅琪队和寰岛队同积7分，名列前茅。")
    labels.append(["O", "O", "B-ORG", "I-ORG", "I-ORG", "O", "B-ORG", "I-ORG", "I-ORG", "O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"])
    texts.append("长梁山地处偏僻，山高路险，冬天零下36摄氏度，吃水都困难。")
    labels.append(["B-LOC", "I-LOC", "I-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"])
    return texts, labels

def load_sentences():
    texts = load_cws_sentences()
    texts.extend(load_ner_sentences()[0])
    return texts
