# -*- coding: utf8 -*-
"""
This is a sample script to construct spaCy 3.0 training data from spaCy 2.0 format

Reference:
* https://towardsdatascience.com/using-spacy-3-0-to-build-a-custom-ner-model-c9256bea098
"""

import pandas as pd
from spacy.tokens import DocBin
from spacy.training import Example
from tqdm import tqdm
import codecs
import datetime
import random
import spacy

if __name__ == '__main__':
    ''' Configurations '''
    LANGUAGE_TYPE = "zh"
    TRAIN_DATA_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/duie_train.csv"
    EPOCH = 20
    SHUFFLE = False
    MODEL_SAVE_PATH = "../../../KnowledgeGraph_materials/model_kg/210506_spacy_model"
    spacy.prefer_gpu()
    '''
    # following is spaCy 2.0 training data format
    TRAIN_DATA = [('The F15 aircraft uses a lot of fuel', {'entities': [(4, 7, 'aircraft')]}),
                  ('did you see the F16 landing?', {'entities': [(16, 19, 'aircraft')]}),
                  ('how many missiles can a F35 carry', {'entities': [(24, 27, 'aircraft')]}),
                  ('is the F15 outdated', {'entities': [(7, 10, 'aircraft')]}),
                  ('does the US still train pilots to dog fight?',
                   {'entities': [(0, 0, 'aircraft')]}),
                  ('how long does it take to train a F16 pilot',
                   {'entities': [(33, 36, 'aircraft')]}),
                  ('how much does a F35 cost', {'entities': [(16, 19, 'aircraft')]}),
                  ('would it be possible to steal a F15', {'entities': [(32, 35, 'aircraft')]}),
                  ('who manufactures the F16', {'entities': [(21, 24, 'aircraft')]}),
                  ('how many countries have bought the F35',
                   {'entities': [(35, 38, 'aircraft')]}),
                  ('is the F35 a waste of money', {'entities': [(7, 10, 'aircraft')]})]
    '''

    ''' Process Starts '''
    data_import = codecs.open(TRAIN_DATA_PATH, mode="r", encoding="utf8", errors="ignore")
    TRAIN_DATA = []
    for lineIndex, line in enumerate(data_import.readlines()[:100]):
        if lineIndex == 0:
            continue

        text = ""
        entity_list = []

        lineElementList = line.split("●")
        for splitIndex, splitElement in enumerate(lineElementList):
            if splitIndex == 0:
                text = splitElement
            else:
                start_index = 999
                end_index = 999
                for stringIndex, char in enumerate(lineElementList[0]):
                    if lineElementList[0][
                       stringIndex: (stringIndex + len(splitElement.replace("\n", "")))] == splitElement.replace("\n",
                                                                                                                 ""):
                        start_index = stringIndex
                        end_index = (stringIndex + len(splitElement.replace("\n", "")))
                        break
                # append value into training data list
                TRAIN_DATA.append((lineElementList[0], {'entities': [(start_index, end_index, "RELATION_ENTITY")]}))

    # print(TRAIN_DATA)

    nlp = spacy.load("zh_core_web_trf")

    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    start_training_time = str(datetime.datetime.now())
    print("Based model loaded, start training the model at \n", start_training_time)

    for epoch in range(EPOCH):
        # 随机化训练数据的顺序
        losses = {}
        if SHUFFLE:
            random.shuffle(TRAIN_DATA)
        # 创建批次并遍历
        for batch in spacy.util.minibatch(TRAIN_DATA, size=100):
            for text, annotations in batch:
                # create Example
                doc = nlp.make_doc(text)
                '''
                tackle if our entity does not match the tokenizer
                Reference:
                * https://github.com/explosion/spaCy/discussions/6979
                * https://github.com/explosion/spaCy/issues/5727
                '''
                span = doc.char_span(annotations["entities"][0][0], annotations["entities"][0][1],
                                     label=annotations["entities"][0][2], alignment_mode="expand")
                adjusted_start_char = span.start_char
                adjusted_end_char = span.end_char
                '''  tacking part ended '''
                example = Example.from_dict(doc, annotations)
                # Update the model
                nlp.update([example], losses=losses, drop=0.3)

        print("Epoch:", epoch + 1, "Loss:", losses)

    # 保存模型
    ending_time = str(datetime.datetime.now())
    time_diff = datetime.datetime.strptime(ending_time, datetimeFormat) - \
                datetime.datetime.strptime(start_training_time, datetimeFormat)
    print("Start training the model at \n", start_training_time)
    print("Training finished at \n", datetime.datetime.now())
    print("Time spent:\n", time_diff)
    nlp.to_disk(MODEL_SAVE_PATH)
