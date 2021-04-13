# -*- coding: utf8 -*-
import codecs
import monpa
import os
import pandas as pd
import re

import pandas as pd


def create_user_dict(TOKEN_DIR, threshold=1):

    for fileIndex, fileElement in enumerate(os.listdir(TOKEN_DIR)):
        if fileIndex == 0:
            data_original = pd.read_csv(TOKEN_DIR + fileElement, sep=",")
            data_original = data_original[data_original["Counts"] > threshold]
        else:
            data_toconcat = pd.read_csv(TOKEN_DIR + fileElement, sep=",")
            data_original = pd.concat([data_original, data_toconcat[data_toconcat["Counts"] > threshold]], ignore_index=True)

    entity_list = data_original.iloc[:, 0].tolist()

    # remove blank inside entity element
    for elementIndex, element in enumerate(entity_list):
        entity_list[elementIndex] = str(entity_list[elementIndex]).replace(" ", "")
        entity_list[elementIndex] = str(entity_list[elementIndex]).replace("  ", "")
        entity_list[elementIndex] = str(entity_list[elementIndex]).replace("\"", "")

    data_dict_output = pd.DataFrame({
        "0": entity_list,
        "1": [int(1000000 * data_original.loc[i, "Counts"]) for i in range(len(entity_list))],
        "2": ["UserEntity" for i in range(len(entity_list))],
    })

    data_dict_output.to_csv("../dicts/monpa_entity_dict.txt", encoding="utf8", sep=" ", index=False, header=False)


if __name__ == '__main__':
    # configurations
    TOKEN_DIR = ""
    RESULT_SAVING_DIR = ""
    DATA_IMPORT_DIR = ""
    STOP_WORD_DIR = "../stopwords/"

    # MONPA
    relation_dependencies_possible_list = []
    relation_pos_possible_list = ["VH", "VC", "VJ", "VA"]
    relation_pos_re = "^[V]"
    entity_dependencies_possible_list = []
    entity_pos_possible_list = ["Na", "Nv", "Neu", "Nes", "Nf", "Ng", "Nh", "Neqa", "Nep", "Ncd", "FW", "DE", "LOC",
                                "ORG"]
    entity_pos_re = "^[N]"
    splitter_pos_list = ["COMMACATEGORY", "PERIODCATEGORY"]
    # Common, usually entity
    sentences_splitter = ["，", "。", "！", "!", "？", "；", ";", "："]
    bracket_entity_list_first = ["(", "（", "[", "［", "{", "｛", "<", "＜", "〔", "【", "〖", "《", "〈"]
    bracket_entity_list_last = [")", "）", "]", "］", "}", "｝", ">", "＞", "〕", "】", "〗", "》", "〉"]
    punct_entity_list = [" " * i for i in range(0, 100)]
    conjuction_entity_list = ["的", "、", "之", "及", "與"]
    not_entity_relation_list = ["的", "、", "之", "及", "與", "\r\n \r\n ", "\r\n \r\n  "] + \
                               [" " * i for i in range(0, 100)] + ["\n" * i for i in range(0, 100)]

    # process starts
    create_user_dict(TOKEN_DIR, 10)
    monpa.load_userdict("../dicts/monpa_entity_dict.txt")

    # read in stop word list
    stopword_list = []
    for fileIndex, fileElement in enumerate(os.listdir("../stopwords/")):
        if fileElement[-2:] != "md":
            data_temp = pd.read_csv("../stopwords/" + fileElement, encoding="utf8", sep="@#$%&*")
            stopword_list += data_temp.iloc[:, 0].tolist()

    stopword_list = list(dict.fromkeys(stopword_list))

    for fileIndex, fileElement in enumerate(os.listdir(DATA_IMPORT_DIR)):
        file = codecs.open(DATA_IMPORT_DIR + fileElement, 'r', encoding='utf8', errors='ignore')
        fileExport = codecs.open(RESULT_SAVING_DIR + fileElement, 'w', encoding='utf8', errors='ignore')

        for textIndex, textElement in enumerate(file):
            # shorten text sentences
            pattern = "|".join(sentences_splitter)
            textList = re.split(pattern, textElement)
            # start pseg text
            for subIndex, subElement in enumerate(textList):
                result_list = monpa.cut(subElement)
                if len(result_list) > 0:
                    for resultIndex, resultElement in enumerate(result_list):
                        if resultElement not in stopword_list and len(resultElement) <= 6:
                            fileExport.write(str(resultElement) + " ")
                    fileExport.write("\n")
                else:
                    continue