# -*- coding: utf8 -*-
"""
ttps://radimrehurek.com/gensim/models/word2vec.html
https://github.com/RaRe-Technologies/gensim-data
"""
from gensim.models import word2vec
import gensim.downloader as api
import warnings

if __name__ == '__main__':
    # configurations
    MODEL_PATH = "../../../models_kg/210415_gensim_CBOW_2.model"

    ''' Process Starts '''
    warnings.filterwarnings("ignore")
    print("Loading model...")
    # model = api.load("conceptnet-numberbatch-17-06-300")
    # model = api.load("glove-twitter-25")
    model = word2vec.Word2Vec.load(MODEL_PATH)
    print("Model loaded.")
    # print(model.most_similar("cat"))

    entityA = "清朝"
    entityB = "中華民國"
    print(entityA, entityB)
    print(model.predict_output_word([entityA, entityB], topn=10))
