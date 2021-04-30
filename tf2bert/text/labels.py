import itertools
import collections

# 与标签相关的处理，序列标注中常用标签的转换

class TaggingTokenizer:
    """BIOES、BIO、BMES标签映射，标签的转换和逆转换"""

    def fit(self, tags):
        self.counter = collections.Counter(itertools.chain(*tags))
        self.labels = sorted(set(self.counter))
        self.id2label = {i:j for i,j in enumerate(self.labels)}
        self.label2id = {j:i for i,j in self.id2label.items()}

    def show(self, density=True):
        c = dict(self.counter)
        if density:
            total = sum(c.values())
            c = {i:j/total for i,j in c.items()}
        return c

    def encode(self, tags):
        """标签转成类别ID"""
        return [self.label2id[tag] for tag in tags]

    def batch_encode(self, batch_tags):
        return [self.encode(tags) for tags in batch_tags]

    def decode(self, ids):
        """类别ID转标签"""
        return [self.id2label[i] for i in ids]

    def batch_decode(self, batch_ids):
        return [self.decode(ids) for ids in batch_ids]

    def id_to_label(self, i):
        return self.id2label[i]

    def label_to_id(self, label):
        return self.label2id[label]

    @property
    def num_classes(self):
        return len(self.labels)

def bio2iobes(tags):
    """BIO标签转IOBES标签"""
    def split_spans(tags):
        buf = []
        for tag in tags:
            if tag == "O" or tag.startswith("B"):
                if buf:
                    yield buf
                buf = [tag]
            else:
                # == tag.startswith("I")
                buf.append(tag)
        if buf:
            yield buf

    ntags = []
    for span in split_spans(tags):
        tag = span[0]
        if len(span) == 1:
            if tag == "O":
                ntags.append(tag)
            else:
                tag = "S" + tag[1:]
                ntags.append(tag)
        else:
            btag = "B" + tag[1:]
            itag = "I" + tag[1:]
            etag = "E" + tag[1:]
            span_tags = [btag] + [itag] * (len(span) - 2) + [etag]
            ntags.extend(span_tags)
    return ntags

def batch_bio2iobes(batch_tags):
    return [bio2iobes(tags) for tags in batch_tags]

def iobes2bio(tags):
    """IOBES标签转BIO标签"""
    ntags = []
    for tag in tags:
        if tag == "O":
            ntags.append(tag)
            continue
        tag, label = tag.split("-")
        if tag == "E":
            tag = "I"
        if tag == "S":
            tag = "B"
        tag = tag + "-" + label
        ntags.append(tag)
    return ntags

def batch_iobes2bio(batch_tags):
    return [iobes2bio(tags) for tags in batch_tags]

def bmes2iobes(tags):
    """BMES标签转成IOBES标签，实则就是 M -> I"""
    ntags = []
    for tag in tags:
        if tag.startswith("M"):
            tag = "I" + tag[1:]
        ntags.append(tag)
    return ntags

def batch_bmes2iobes(batch_tags):
    return [bmes2iobes(tags) for tags in batch_tags]

def bmes2bio(tags):
    iobes_tags = bmes2iobes(tags)
    bio_tags = iobes2bio(iobes_tags)
    return bio_tags

def batch_bmes2bio(batch_tags):
    return [bmes2bio(tags) for tags in batch_tags]

def words2bmes(words, sep=" "):
    """词序列转BMES标签"""
    if isinstance(words, str):
        words = words.split(sep)
    tags = []
    for word in words:
        if not word:
            raise ValueError("None or zero-length word from {}".format(str(words)))
        if len(word) == 1:
            tags.append("S")
        else:
            tags.extend(["B"] + ["M"] * (len(word) - 2) + ["E"])
    return tags

def batch_words2bmes(batch_words, sep=" "):
    return [words2bmes(words, sep) for words in batch_words]

def bmes2words(text, tags):
    """根据BMES标签从text中切分出词"""
    chars = list(text)
    results = []
    if len(chars) == 0:
        return results

    word = chars[0]
    for char, tag in zip(chars[1:], tags[1:]):
        if tag in ("B", "S"):
            results.append(word)
            word = ""
        word += char

    if len(word) != 0:
        results.append(word)
    return results

def batch_bmes2words(batch_text, batch_tags):
    return [bmes2words(text, tags) for text, tags in zip(batch_text, batch_tags)]

def check_tags(tags, fix=False):
    """检测tags序列是否符合规则"""

def find_entities(text, tags, withO=False):
    """根据标签提取文本中的实体，适合BIO和BIOES标签，
    withO是否返回O标签内容。
    """
    def segment_by_tags(text, tags):
        buf = ""
        plabel = None
        for tag, char in zip(tags, text):
            if tag == "O":
                label = tag
            else:
                tag, label = tag.split("-")
            if tag == "B" or tag == "S":
                if buf:
                    yield buf, plabel
                buf = char
                plabel = label
            elif tag == "I" or tag == "E":
                buf += char
            elif withO and tag == "O":
                # tag == "O"
                if buf and plabel != "O":
                    yield buf, plabel
                    buf = ""
                buf += char
                plabel = label
        if buf:
            yield buf, plabel
    return list(segment_by_tags(text, tags))

def find_entities_chunking(tags):
    """根据标签提取文本中的实体始止位置，
    兼容BIO和BIOES标签。
    """
    def chunking_by_tags(tags):
        buf = None
        plabel = None
        for i, tag in enumerate(tags):
            if tag == "O":
                label = tag
            else:
                tag, label = tag.split("-")
            if tag == "B" or tag == "S":
                if buf:
                    yield (plabel, *buf)
                buf = [i, i+1]
                plabel = label
            elif tag == "I" or tag == "E":
                buf[1] += 1
        if buf:
            yield (plabel, *buf)
    return list(chunking_by_tags(tags))

def batch_find_entities_chunking(batch_tags):
    """批量操作的find_entities_chunking"""
    return [find_entities_chunking(tags) for tags in batch_tags]

def find_words(text, tags):
    """通过SBME序列对text分词"""
    def segment_by_tags(text, tags):
        buf = ""
        for tag, char in zip(tags, text):
            # t is S or B
            if tag in ("B", "S"):
                if buf:
                    yield buf
                buf = char
            # t is M or E
            else:
                buf += char
        if buf:
            yield buf
    return list(segment_by_tags(text, tags))
