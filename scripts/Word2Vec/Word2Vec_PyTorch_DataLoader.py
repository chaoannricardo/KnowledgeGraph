# -*- coding: utf8 -*-
"""
PyTorch DataLoaders Documentation: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
Word2Vec DataLoaders Reference: https://cloud.tencent.com/developer/article/1613950
"""
import torch
import torch.utils.data as tud


class Word2Vec_PyTorch_Embedding(tud.Dataset):
    def __init__(self, text, word2idx, idx2word, word_freqs, word_counts,  context_window, negative_sample_count):
        """ text: a list of words, all text from the training dataset
            word2idx: the dictionary from word to index
            idx2word: index to word mapping
            word_freqs: the frequency of each word
            word_counts: the word counts

            torch.Tensor & LongTensor documentations:
            * https://pytorch.org/docs/stable/tensors.html
            * https://newpower.tistory.com/199
        """
        # super methods from parent class and reconstruct the methods
        super(Word2Vec_PyTorch_Embedding, self).__init__()
        self.text_encoded = [word2idx.get(word, word2idx["<UNK>"]) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)
        self.context_window = context_window
        self.negative_sample_count = negative_sample_count

    def __len__(self):
        # return length of the words encoded
        return len(self.text_encoded)

    def __getitem__(self, idx):
        """
        The function create following datas for training
        * center word
        * positive word near the center word
        * random K word as negative samples
        :return:
        """
        center_words = self.text_encoded[idx]
        pos_indices = list(range(idx - self.context_window, idx)) + list(range(idx + 1, idx + 1 + self.context_window))
        # preventing the indices fall outside the boundary
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]

        neg_words = torch.multinomial(self.word_freqs, self.negative_sample_count * pos_words.shape[0])

        return center_words, pos_words, neg_words




