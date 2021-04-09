# -*- coding: utf8 -*-
import os
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import codecs
import math
import numpy as np
import pandas as pd
import random
import scipy
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

'''
Word2Vec Implementation with Pytorch
Reference: https://cloud.tencent.com/developer/article/1613950
'''


def import_data(data_path, vocab_size):
    with codecs.open(data_path, "r", encoding="utf8", errors="ignore") as file:
        text_total = file.read()
    text_total = text_total.split()
    text_total = list(filter((["\n"]).__ne__, text_total))
    # construct vocabulary dictionary
    vocab_dict = dict(Counter(text_total).most_common(vocab_size - 1))
    vocab_dict["<UNK>"] = len(text_total) - np.sum(len(list(vocab_dict.values())))
    idx2word = [word for word in vocab_dict.keys()]
    word2index = {word:index for index, word in enumerate(vocab_dict.keys())}
    word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)
    word_freqs = (word_counts / np.sum(word_counts)) ** 0.75
    return text_total, vocab_dict, idx2word, word2index, word_counts, word_freqs


if __name__ == '__main__':
    '''
    Parameters
    '''
    RANDOM_SEED = 1
    CONTEXT_WINDOW = 3
    NEGATIVE_SAMPLE_COUNT = 5
    EPOCHS = 2
    MAX_VOCAB_SIZE = 10000
    BATCH_SIZE = 32
    LR = 0.2
    DATA_PATH = ""
    '''
    Main Process Starts
    '''
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    





















