import random
from tf2bert.text.collections import Trie, _sentinel
from tf2bert.text.utils import load_cws_sentences

def load_words():
    import jieba
    words = []
    for text in load_cws_sentences():
        words.extend(jieba.lcut(text))
    return words

def test_trie():
    trie = Trie()
    words = load_words()
    words = random.sample(words, k=100)
    trie.update(words)
    trie.update(["广东省", "长假", "成绩单"])
    assert "广东省" in trie

    trie.pop("成绩单")
    assert "成绩单" not in trie

    assert trie["长假"] is _sentinel

test_trie()
