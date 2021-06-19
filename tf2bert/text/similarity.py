import numpy as np
from scipy import optimize
from scipy.spatial import distance
from tf2bert.text.dp import min_edit_distance

# 简单的文本相似性计算
# 1. 词向量序列平均后余弦相似
# 2. tfidf加权平均余弦相似
# 3. wmd
# 4. jaccard
# 5. editdistance

def cosine_similar(v1, v2):
    """余弦相似度，取值[0,1]之间，值越大越相似"""
    return 1 - distance.cosine(v1, v2)

def tfidf_cosine_similar(text1, text2):
    pass

def idf_cosine_similar(text1, text2):
    pass

def jaccard(text1, text2):
    """jaccard距离"""
    s1 = set(text1)
    s2 = set(text2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def similar_min_editdistance(text1, text2):
    distance = min_edit_distance(text1, text2)
    return 1 - distance / max(len(text1), len(text2))

def wasserstein_distance(p, q, C):
    """Wasserstein距离计算方法，
    p.shape=(m,)
    q.shape=(n,)
    C.shape=(m,n)
    p q满足归一性概率化
    """
    p = np.array(p)
    q = np.array(q)
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(C)
        A[i,:] = 1.0
        A_eq.append(A.reshape((-1,)))
    for i in range(len(q)):
        A = np.zeros_like(C)
        A[:,i] = 1.0
        A_eq.append(A.reshape((-1,)))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q], axis=0)
    C = np.array(C).reshape((-1,))
    return optimize.linprog(
        c=C,
        A_eq=A_eq[:-1],
        b_eq=b_eq[:-1],
        method="interior-point",
        options={"cholesky":False, "sym_pos":True}
    ).fun

def word_mover_distance(x, y):
    """Word Mover's Distance计算方法, 
    x.shape=(m,d)
    y.shape=(n,d)
    """
    x = np.array(x)
    y = np.array(y)
    p = np.ones(x.shape[0]) / x.shape[0]
    q = np.ones(y.shape[0]) / y.shape[0]
    C = np.sqrt(np.mean(np.square(x[:,None] - y[None,:]), axis=2))
    return wasserstein_distance(p, q, C)

if __name__ == "__main__":
    x = np.random.uniform(size=(64, 12))
    y = np.random.normal(size=(64, 12))
    print(word_mover_distance(x, y))
