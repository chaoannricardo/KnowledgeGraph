# -*- coding: utf8 -*-
"""
Word2Vec DataLoaders Reference: https://cloud.tencent.com/developer/article/1613950
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        """
        input_labels: center words, [batch_size]
        pos_labels: positive words, [batch_size, (window_size * 2)]
        neg_labels：negative words, [batch_size, (window_size * 2 * K)]
        return: loss, [batch_size]

        Reference:
        torch.unsqueeze & squeeze reference: https://blog.csdn.net/xiexu911/article/details/80820028
        torch.bmm reference: https://pytorch.org/docs/stable/generated/torch.bmm.html

        """
        input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]
        pos_embedding = self.in_embed(pos_labels)  # [batch_size, (window * 2), embed_size]
        neg_embedding = self.in_embed(neg_labels)  # [batch_size, (window * 2 * K), embed_size]

        input_embedding = torch.unsqueeze(input_embedding, 2)  # [batch_size, embed_size, 1]

        pos_dot = torch.bmm(pos_embedding, input_embedding)  # [batch_size, (window * 2), 1]
        pos_dot = torch.squeeze(pos_dot, 2)  # [batch_size, (window * 2)]

        neg_dot = torch.bmm(neg_embedding, -input_embedding) # [batch_size, (window * 2 * K), 1]
        neg_dot = torch.squeeze(neg_dot, 2)

        log_pos = torch.sum(F.logsigmoid(pos_dot), 1)
        log_neg = torch.sum(F.logsigmoid(neg_dot), 1)

        loss = log_pos + log_neg

        return -loss

    def input_embedding(self):
        return self.in_embed.weight.numpy()


if __name__ == '__main__':
    pass
