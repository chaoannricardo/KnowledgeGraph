# -*- coding: utf8 -*-
"""
ttps://radimrehurek.com/gensim/models/word2vec.html
https://github.com/RaRe-Technologies/gensim-data
"""
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

if __name__ == '__main__':
    print("Loading model...")
    model = api.load("conceptnet-numberbatch-17-06-300")
    # model = api.load("glove-twitter-25")
    print("Model loaded.")
    print(model.most_similar("cat"))
    print(model.predict_output_word(["Obama", "Hawaii"], topn=10))
