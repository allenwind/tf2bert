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
    texts.append("The quick brown fox jumps over the lazy dog.")
    texts.append("人的复杂的生理系统的特性注定了一件事情，就是从懂得某个道理到执行之间，是一个漫长的回路。")
    texts.append("除了导致大批旅客包括许多准备前往台北采访空难的新闻记者滞留在香港机场，直到下午2:17分日本亚细亚航空公司开出第一班离港到台北的班机才疏导了滞留在机场的旅客。")
    return texts
