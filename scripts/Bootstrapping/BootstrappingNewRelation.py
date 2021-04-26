# -*- coding: utf8 -*-
from tqdm import tqdm
import codecs
import networkx as nx
import re
import spacy

if __name__ == '__main__':
    ''' Configurations '''
    BASIC_SEED_PATH = ""  # seed dictionary constructed by former script
    DATA_IMPORT_PATH = ""  # data used to find new relations
    OBJECT_DICT_PATH = ""
    SUBJECT_DICT_PATH = ""
    TRIGGER_WORD_PATH = ""
    NEW_SEED_OUTPUT_PATH = ""
    NEW_BASIC_SEED_OUTPUT_PATH = ""
    NEW_WHOLE_SEED_OUTPUT_PATH = ""
    NEW_TRIGGER_WORD_OUTPUT_PATH = ""
    SPACY_ENGINE_TYPE = "zh_core_web_trf"  # "zh_core_web_sm" "en_core_web_sm"
    ITERATIONS = 10

    ''' Process Starts '''