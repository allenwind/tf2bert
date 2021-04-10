import time
import random
import glob
import queue
import collections
import heapq
from functools import wraps
from operator import itemgetter
from multiprocessing import Pool, Queue
from concurrent.futures import ThreadPoolExecutor

# from https://github.com/allenwind/count-in-parallel

basic_tokenize = lambda text: list(text)

def load_batch_texts(files, batch_size=300, limit=None, shuffle=True):
    # 批量的形式加载文本

    def load(file):
        with open(file, "r", encoding="utf-8") as fd:
            text = fd.read()
        return text

    if not files:
        raise FileNotFoundError("without any files")

    if shuffle:
        random.shuffle(files)

    files = files[:limit]
    executor = ThreadPoolExecutor(max_workers=1)
    gen = executor.map(load, files)
    yield from split_into_batchs(gen, batch_size)

def split_into_batchs(gen, batch_size=300):
    batch_texts = []
    for text in gen:
        batch_texts.append(text)
        if len(batch_texts) == batch_size:
            yield batch_texts
            batch_texts = []
    if batch_texts:
        yield batch_texts

def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, "elapsed time {:.3f}s".format(end-start))
        return result
    return wrapper

class Counter(collections.Counter):

    def __missing__(self, key):
        return 0

    def most_common(self, n=None):
        # 支持传入负数，表示least common
        if n is None:
            return sorted(self.items(), key=itemgetter(1), reverse=True)
        if n > 0:
            return heapq.nlargest(n, self.items(), key=itemgetter(1))
        return self.least_common(-n)

    def least_common(self, n):
        # 最不常见
        return heapq.nsmallest(n, self.items(), key=itemgetter(1))

@timethis
def count_in_parallel(
    tokenize,
    batch_generator,
    processes,
    maxsize=300,
    preprocess=None):
    # 文本tokenize前的预处理
    if preprocess is None:
        preprocess = lambda x: x

    def batch_counter(batch_texts_queue, tokens_queue):
        # 批量统计
        while True:
            tokens = Counter()
            batch_texts = batch_texts_queue.get()
            for text in batch_texts:
                text = preprocess(text)
                for token in tokenize(text):
                    tokens[token] += 1
            tokens_queue.put(tokens)

    # 数据队列
    batch_texts_queue = Queue(maxsize)
    tokens_queue = Queue()
    
    # 进程池
    pool = Pool(processes, batch_counter, initargs=(batch_texts_queue, tokens_queue))

    # 全局统计表
    gtokens = Counter()
    def merge_tokens():
        # 合并每个进程的统计表
        batch_tokens_size = 0
        for _ in range(tokens_queue.qsize()):
            tokens = tokens_queue.get()
            batch_tokens_size += 1
            for k, v in tokens.items():
                gtokens[k] += v
        return batch_tokens_size

    batch_tokens_size = 0
    for batch_texts_size, batch_texts in enumerate(batch_generator, start=1):
        while True:
            try:
                batch_texts_queue.put(batch_texts, block=False)
                break
            except queue.Full:
                batch_tokens_size += merge_tokens()

            if batch_texts_size % (processes * maxsize // 2) == 0:
                batch_tokens_size += merge_tokens()

    while batch_tokens_size != batch_texts_size:
        batch_tokens_size += merge_tokens()

    pool.terminate()
    return gtokens

def count_in_parallel_from_files(
    tokenize,
    files,
    processes,
    maxsize=300,
    preprocess=None,
    batch_size=300,
    limit=None,
    shuffle=True):
    batch_generator = load_batch_texts(files, batch_size, limit, shuffle)
    tokens = count_in_parallel(
        tokenize,
        batch_generator,
        processes,
        maxsize,
        preprocess
    )
    return tokens

def count_in_parallel_from_generator(
    tokenize,
    generator,
    processes,
    maxsize=300,
    preprocess=None,
    batch_size=300):
    batch_generator = split_into_batchs(generator, batch_size)
    tokens = count_in_parallel(
        tokenize,
        batch_generator,
        processes,
        maxsize,
        preprocess
    )
    return tokens

def batch_generator(generator, batch_size=300):
    batch_texts = []
    for i, text in enumerate(generator):
        batch_texts.append((i, text))
        if len(batch_texts) == batch_size:
            yield batch_texts
            batch_texts = []
    if batch_texts:
        yield batch_texts

@timethis
def tokenize_in_parallel(
    tokenize,
    generator,
    processes=7,
    maxsize=300,
    preprocess=None):
    if preprocess is None:
        preprocess = lambda x: x

    def batch_tokenize(batch_texts_queue, tokens_queue):
        while True:
            batch_tokens = []
            batch_texts = batch_texts_queue.get()
            for i, text in batch_texts:
                text = preprocess(text)
                tokens = tokenize(text)
                batch_tokens.append((i, tokens))
            tokens_queue.put(batch_tokens)

    batch_texts_queue = Queue(maxsize)
    tokens_queue = Queue()
    pool = Pool(processes, batch_tokenize, initargs=(batch_texts_queue, tokens_queue))

    gtokens = []
    def merge():
        batch_tokens_size = 0
        for _ in range(tokens_queue.qsize()):
            batch_tokens = tokens_queue.get()
            batch_tokens_size += 1
            for i, tokens in batch_tokens:
                gtokens.append((i, tokens))
        return batch_tokens_size

    batch_tokens_size = 0
    for batch_texts_size, batch_texts in enumerate(batch_generator(generator), start=1):
        while True:
            try:
                batch_texts_queue.put(batch_texts, block=False)
                break
            except queue.Full:
                batch_tokens_size += merge()

            if batch_texts_size % (processes * maxsize // 2) == 0:
                batch_tokens_size += merge()

    while batch_tokens_size != batch_texts_size:
        batch_tokens_size += merge()

    pool.terminate()
    gtokens.sort(key=lambda x: x[0]) # 还原原来的输入顺序
    gtokens = [i[1] for i in gtokens]
    return gtokens
