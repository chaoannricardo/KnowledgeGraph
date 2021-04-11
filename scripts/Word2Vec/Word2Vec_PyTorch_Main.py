# -*- coding: utf8 -*-
"""
Word2Vec Implementation with Pytorch
Code Reference: https://cloud.tencent.com/developer/article/1613950

Other related information:
Pytorch Embedding Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
Pytorch Tensor: https://pytorch.org/docs/stable/tensors.html
Pytorch Optimizer: https://pytorch.org/docs/stable/optim.html
Optimizer Reference: https://mlfromscratch.com/optimizers-explained/#/
Batch Size Impact: https://medium.com/deep-learning-experiments/effect-of-batch-size-on-neural-net-training-c5ae8516e57

"""
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
import torch.optim as optim

# import user defined packages
import Word2Vec_PyTorch_DataLoader
import Word2Vec_PyTorch_Model


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
    EMBEDDING_SIZE = 1000
    BATCH_SIZE = 32
    LR = 0.2
    DATA_PATH = "../../results/210408_result/210408_token_sentences_for_word2vec.csv"
    '''
    Main Process Starts
    '''
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    text_total, vocab_dict, idx2word, word2index, word_counts, word_freqs = import_data(DATA_PATH, MAX_VOCAB_SIZE)
    dataset = Word2Vec_PyTorch_DataLoader.Word2Vec_PyTorch_Embedding(text=text_total,
                                                                     word2idx=word2index,
                                                                     idx2word=idx2word,
                                                                     word_freqs=word_freqs,
                                                                     word_counts=word_counts,
                                                                     context_window=CONTEXT_WINDOW,
                                                                     negative_sample_count=NEGATIVE_SAMPLE_COUNT)
    dataloader = tud.DataLoader(dataset=dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = Word2Vec_PyTorch_Model.EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    for e in range(EPOCHS):
        for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
            input_labels = input_labels.long()
            pos_labels = pos_labels.long()
            neg_labels = neg_labels.long()

            optimizer.zero_grad()
            loss = model.forward(input_labels=input_labels,
                                 pos_labels=pos_labels,
                                 neg_labels=neg_labels).mean()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("Epoch ", (e + 1), " Iteration ", i, loss.item())

    embedding_weights = model.input_embedding()
    torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))
































