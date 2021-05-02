# -*- coding: utf8 -*-
"""
This is a sample script to construct spaCy 3.0 training data from spaCy 2.0 format

Reference:
* https://towardsdatascience.com/using-spacy-3-0-to-build-a-custom-ner-model-c9256bea098
"""

import pandas as pd
from spacy.tokens import DocBin
from tqdm import tqdm
import codecs
import spacy

if __name__ == '__main__':
    ''' Configurations '''
    LANGUAGE_TYPE = "zh"
    TRAIN_DATA_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/duie_train.csv"
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
                    if lineElementList[0][stringIndex: (stringIndex + len(splitElement.replace("\n", "")))] == splitElement.replace("\n", ""):
                        start_index = stringIndex
                        end_index = (stringIndex + len(splitElement.replace("\n", "")))
                        break
                # append value into training data list
                TRAIN_DATA.append((lineElementList[0], {'entities': [(start_index, end_index, "RELATION_ENTITY")]}))

    # print(TRAIN_DATA)

    ''' Process to Construct spaCy 3.0 trainging data format '''
    nlp = spacy.blank(LANGUAGE_TYPE)  # load a new spacy model
    db = DocBin()  # create a DocBin object

    for text, annot in tqdm(TRAIN_DATA):  # data in previous format
        doc = nlp.make_doc(text)  # create doc object from text
        ents = []
        for start, end, label in annot["entities"]:  # add character indexes
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        doc.ents = ents  # label the text with the ents
        db.add(doc)

    db.to_disk("./train.spacy")  # save the docbin object
