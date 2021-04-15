# -*- coding: utf8 -*-
"""
ttps://radimrehurek.com/gensim/models/word2vec.html
https://github.com/RaRe-Technologies/gensim-data
"""
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

if __name__ == '__main__':
    model = api.load("conceptnet-numberbatch-17-06-300")
    model.predict_output_word(["Obama", "Hawaii"], topn=10)

