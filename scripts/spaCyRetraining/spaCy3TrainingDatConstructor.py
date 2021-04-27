# -*- coding: utf8 -*-
"""
This is a script to construct spaCy 3.0 training data from spaCy 2.0 format

Reference:
* https://towardsdatascience.com/using-spacy-3-0-to-build-a-custom-ner-model-c9256bea098
"""

import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin

if __name__ == '__main__':
    nlp = spacy.blank("en")  # load a new spacy model
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