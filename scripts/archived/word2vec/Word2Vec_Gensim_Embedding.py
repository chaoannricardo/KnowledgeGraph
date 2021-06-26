# -*- coding: utf8 -*-
from gensim.models import word2vec
from monpa import utils
from tqdm import tqdm
import codecs
import logging
import monpa
import numpy as np
import os
import pandas as pd
import shutil
import spacy
import warnings

if __name__ == '__main__':
    # configurations
    MODEL_SAVING_PATH = "../../../models_kg/210415_gensim_CBOW_2.model"
    TRAINING_DATA_DIR = "../../../results_kg/210415_result/word2vec_dataset/"


    # process starts
    warnings.filterwarnings("ignore")
    # fileExport = open("./gensim_temp.txt", mode="w", encoding="utf8")
    #
    # for fileIndex, fileElement in enumerate(os.listdir(TRAINING_DATA_DIR)):
    #     fileImport = open(TRAINING_DATA_DIR + fileElement, mode="r", encoding="utf8", errors="ignore")
    #     shutil.copyfileobj(fileImport, fileExport)
    #     fileImport.close()

    sentences = word2vec.LineSentence("./gensim_temp.txt")

    # https://radimrehurek.com/gensim/models/word2vec.html
    # parameter tuning 1: https://datascience.stackexchange.com/questions/51404/word2vec-how-to-choose-the-embedding-size-parameter
    # parameter tuning 2: https://stackoverflow.com/questions/29939984/word2vec-and-gensim-parameters-equivalence
    print("Strating to train the model...")
    model = word2vec.Word2Vec(sentences=sentences, corpus_file=None, vector_size=5000, alpha=0.025,
                              window=1, min_count=5, max_vocab_size=None, sample=0.001,
                              seed=1, workers=16, min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75,
                              cbow_mean=1, epochs=100, null_word=0, trim_rule=None, sorted_vocab=1,
                              batch_words=1000, compute_loss=False, callbacks=(),  max_final_vocab=None)

    model.save(MODEL_SAVING_PATH)
