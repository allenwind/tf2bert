import numpy as np

def max_sub_array(array):
    pass

def viterbi_decode(scores, trans, return_score=False):
    """使用viterbi算法求最优路径，
    scores.shape = (seq_len, num_tags)
    trans.shape = (num_tags, num_tags)
    """
    dp = np.zeros_like(scores)
    backpointers = np.zeros_like(scores, dtype=np.int32)
    dp[0] = scores[0]
    for t in range(1, scores.shape[0]):
        # 状态转移分值
        v = np.expand_dims(dp[t-1], axis=1) + trans
        dp[t] = scores[t] + np.max(v, axis=0)
        # 记录上一时刻概率最大结点
        backpointers[t] = np.argmax(v, axis=0)

    viterbi = [np.argmax(dp[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    if return_score:
        viterbi_score = np.max(dp[-1])
        return viterbi, viterbi_score
    return viterbi

def longest_common_prefix(texts):
    """最长公共前缀"""
    if not texts:
        return ""
    text1 = min(texts)
    text2 = max(texts)
    for i, s in enumerate(text1):
        if s != text2[i]:
            return text2[:i]
    return text1

def longest_common_substring(text1, text2):
    """最长公共子字符串，区分大小写"""
    n = len(text1)
    m = len(text2)
    maxlen = 0
    span1 = (0, 0)
    span2 = (0, 0)
    if n * m == 0:
        return span1, span2, maxlen

    dp = np.zeros((n+1, m+1), dtype=np.int32)
    for i in range(1, n+1):
        for j in range(1, m+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > maxlen:
                    maxlen = dp[i][j]
                    span1 = (i - maxlen, i)
                    span2 = (j - maxlen, j)       
    return span1, span2, maxlen

def longest_common_subsequence(text1, text2):
    """最长公共子序列"""
    n = len(text1)
    m = len(text2)
    maxlen = 0
    spans1 = []
    spans2 = []
    if n * m == 0:
        return spans1, spans2, maxlen

    dp = np.zeros((n+1, m+1), dtype=np.int32)
    for i in range(1, n+1):
        for j in range(1, m+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    maxlen = dp[-1][-1]

    i = n - 1
    j = m - 1
    while len(spans1) < maxlen:
        if text1[i] == text2[j]:
            spans1.append(i)
            spans2.append(j)
            i -= 1
            j -= 1
        elif dp[i+1, j] > dp[i,j+1]:
            j -= 1
        else:
            i -= 1
    spans1 = spans1[::-1]
    spans2 = spans2[::-1]
    return spans1, spans2, maxlen

def min_edit_distance(text1, text2):
    """Levenshtein distance"""
    n = len(text1)
    m = len(text2)

    # 空串情况
    if n * m == 0:
        return n + m

    dp = np.zeros((n+1, m+1), dtype=np.int32)

    def I(a, b):
        """指示函数"""
        return 1 if a != b else 0

    # 初始化边界状态
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j

    # 计算所有dp值
    for i in range(1, n+1):
        for j in range(1, m+1):
            left = dp[i-1][j] + 1
            down = dp[i][j-1] + 1
            left_down = dp[i-1][j-1] + I(text1[i-1], text2[j-1])
            dp[i][j] = min(left, down, left_down)
    return int(dp[n][m])

if __name__ == "__main__":
    text1 = "kldfhsoijelaksdjfskkl"
    text2 = "kdfsoijelaksesjjskkls"

    span1, span2, maxlen = longest_common_substring(text1, text2)
    print(text1)
    print(text2)
    print(text1[span1[0]:span1[1]])
    print(text2[span2[0]:span2[1]])
    print(maxlen)

    spans1, spans2, maxlen = longest_common_subsequence(text1, text2)
    print(spans1)
    print(spans2)
    print(maxlen)
    print("".join([text1[i] for i in spans1]))
    print("".join([text2[i] for i in spans2]))

    maxlen = min_edit_distance(text1, text2)
    print(maxlen)
