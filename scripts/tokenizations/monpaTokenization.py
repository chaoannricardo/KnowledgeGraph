# -*- coding: utf8 -*-
from monpa import utils
from tqdm import tqdm
import codecs
import monpa
import numpy as np
import os
import pandas as pd


if __name__ == '__main__':
    # configurations
    RESULT_SAVING_DIR = "../../../results_kg/210415_result/"
    DATA_IMPORT_DIR = "../../../data_kg/data_wikipedia_xml_clean/"
    USER_DICT_PATH = ""
    # process starts
    # monpa.load_userdict(USER_DICT_PATH)
    for fileIndex, fileElement in enumerate(tqdm(os.listdir(DATA_IMPORT_DIR))):
        # import data
        psegResultList = []
        labelList = []
        # open file to import
        file = codecs.open(DATA_IMPORT_DIR + fileElement, 'r', encoding='utf8', errors='ignore')
        # open file to export
        fileExport = codecs.open(RESULT_SAVING_DIR + fileElement, 'w', encoding='utf8', errors='ignore')

        for textIndex, textElement in enumerate(file):
            # shorten text sentences
            textList = utils.short_sentence(textElement)
            # start pseg text
            for textIndex, textElement in enumerate(textList):
                psegResult = monpa.pseg(textElement)
                psegResultList.append(psegResult)

            for listIndex, elementList in enumerate(psegResultList):
                if len(elementList) > 0:
                    for elementIndex, elementTuple in enumerate(elementList):
                        elementTuple_0 = filter("|".__ne__, str(elementTuple[0]))
                        fileExport.write(str(elementTuple_0)+","+str(elementTuple[1])+"|")

            fileExport.write("\n")

        file.close()
        fileExport.close()

