import numpy as np
import collections

# ANSI escape code
ansi_code_ids =  [15, 224, 217, 210, 203, 9, 160, 124, 88]
length = len(ansi_code_ids)

hexcolors = [
    "#f5f5ff",
    "#ffe0e0",
    "#ffb6b6",
    "#ff8d8d",
    "#ff6363",
    "#ff3939",
    "#ff1010",
    "#f20000",
    "#dd0000",
    "#c80000",
    "#b40000",
    "#9f0000",
    "#8a0000",
]
hex_lenght = len(hexcolors)

template = "\033[38;5;{value}m{string}\033[0m"
def render_color_text(text, ws):
    ss = []
    ws = np.array(ws)
    ws = (ws - np.min(ws)) / (np.max(ws) - np.min(ws)) * 0.99
    for string, w in zip(text, ws):
        vid = int(w * length)
        value = ansi_code_ids[vid]
        s = template.format(string=string, value=value)
        ss.append(s)
    return "".join(ss)

def print_color_text(text, ws):
    ws = np.array(ws)
    ws = (ws - np.min(ws)) / (np.max(ws) - np.min(ws)) * 0.99
    for string, w in zip(text, ws):
        vid = int(w * length)
        value = ansi_code_ids[vid]
        print(template.format(string=string, value=value), end="")
    print()

markdown_template = '<font color="{}">{}</font>'
def render_color_markdown(text, ws):
    ws = np.array(ws)
    ws = (ws - np.min(ws)) / (np.max(ws) - np.min(ws)) * 0.99
    ss = []
    for string, w in zip(text, ws):
        i = int(w * hex_lenght)
        ss.append(markdown_template.format(hexcolors[i], string))
    return "".join(ss)

def print_match_substring(text1, text2, reverse_render=False):
    # 高亮两个文本的最长公共子串
    if not (text1 and text2):
        return
    s1 = len(text1)
    s2 = len(text2)
    dp = np.zeros((s1 + 1, s2 + 1), dtype=np.int32)
    maxlen = 0
    span1 = (0, 0)
    span2 = (0, 0)
    for i in range(1, s1 + 1):
        for j in range(1, s2 + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > maxlen:
                    maxlen = dp[i][j]
                    span1 = (i - maxlen, i)
                    span2 = (j - maxlen, j)

    w1 = np.zeros(s1)
    w2 = np.zeros(s2)
    w1[span1[0]:span1[1]] = 1
    w2[span2[0]:span2[1]] = 1
    if reverse_render:
        w1 = 1 - w1
        w2 = 1 - w2

    print_color_text(text1, w1)
    print_color_text(text2, w2)

def print_match_subsequence(text1, text2, reverse_render=False):
    # 高亮两个文本的最长公共子序列（LCS）
    if not (text1 and text2):
        return
    s1 = len(text1)
    s2 = len(text2)
    dp = np.zeros((s1 + 1, s2 + 1), dtype=np.int32)
    for i in range(1, s1 + 1):
        for j in range(1, s2 + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    maxlen = dp[-1][-1]

    i = s1 - 1
    j = s2 - 1
    locs = []
    while len(locs) < maxlen:
        if text1[i] == text2[j]:
            locs.append((i, j))
            i -= 1
            j -= 1
        elif dp[i + 1, j] > dp[i, j + 1]:
            j -= 1
        else:
            i -= 1
    locs[::-1] = locs

    sw1 = np.zeros(s1)
    sw2 = np.zeros(s2)
    ts1 = set([i for i, _ in locs])
    ts2 = set([j for _, j in locs])
    for i, j in enumerate(text1):
        if i in ts1:
            sw1[i] = 1
    for i, j in enumerate(text2):
        if i in ts2:
            sw2[i] = 1

    if reverse_render:
        sw1 = 1 - sw1
        sw2 = 1 - sw2

    print_color_text(text1, sw1)
    print_color_text(text2, sw2)

def print_match_jaccard(text1, text2, reverse_render=False):
    # 逐位置比较差异
    w1 = np.zeros(len(text1), dtype=np.float32)
    w2 = np.zeros(len(text2), dtype=np.float32)
    for i, (c1, c2) in enumerate(zip(text1, text2)):
        if c1 == c2:
            w1[i] = 1.0
            w2[i] = 1.0

    if reverse_render:
        w1 = 1 - w1
        w2 = 1 - w2

    print_color_text(text1, w1)
    print_color_text(text2, w2)

def min_edit_distance(word1, word2):
    # Levenshtein distance
    n = len(word1)
    m = len(word2)

    # 空串情况
    if n * m == 0:
        return n + m

    dp = np.zeros((n+1, m+1), dtype=np.int32)

    # 初始化边界状态
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # 计算所有dp值
    for i in range(1, n+1):
        for j in range(1, m+1):
            left = dp[i-1][j] + 1
            down = dp[i][j-1] + 1
            left_down = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                left_down += 1
            dp[i][j] = min(left, down, left_down)
    return dp[n][m]

if __name__ == "__main__":
    # for testing
    import string
    text = string.ascii_letters
    print_color_text(text, np.arange(len(text)))
    print(render_color_markdown(text, np.arange(len(text))))

    text1 = "NLP的魅力在于不断探索"
    text2 = "NLP的梦魇在于不断调参"
    print_match_subsequence(text1, text2, True)
    print("levenshtein distance:", min_edit_distance(text1, text2))

    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "The lazy brown fox jumps over the quick dog"
    print_match_substring(text1, text2, True)

    print_match_jaccard(text1, text2, True)
