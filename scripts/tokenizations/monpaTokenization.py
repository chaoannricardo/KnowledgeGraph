# -*- coding: utf8 -*-
from monpa import utils
from tqdm import tqdm
import codecs
import monpa
import os
import numpy as np
import pandas as pd
import psutil
import re
import warnings

if __name__ == '__main__':
    # configurations
    warnings.filterwarnings("ignore")
    RESULT_SAVING_DIR = "../../../results_kg/210415_result/"
    DATA_IMPORT_DIR = "../../../data_kg/data_wikipedia_xml_clean/"
    STOP_WORD_DIR = "../stopwords/"
    USER_DICT_PATH = ""
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

    ''' Process Starts '''
    # monpa.load_userdict(USER_DICT_PATH)
    # read in stopwords
    stopword_list = []
    for fileIndex, fileElement in enumerate(os.listdir("../stopwords/")):
        if fileElement[-2:] != "md":
            data_temp = pd.read_csv("../stopwords/" + fileElement, encoding="utf8", sep="@#$%&*")
            stopword_list += data_temp.iloc[:, 0].tolist()

    stopword_list = list(dict.fromkeys(stopword_list))
    # print("停用詞數目：", len(stopword_list), "\n", stopword_list[:10], "\n")

    # creating dataframe to store entities and relations
    data_entities = pd.DataFrame({
        "Tokens": [],
        "Counts": []
    })

    data_relations = pd.DataFrame({
        "Tokens": [],
        "Counts": []
    })

    for fileIndex, fileElement in enumerate(tqdm(os.listdir(DATA_IMPORT_DIR))):
        # open file to import
        file = codecs.open(DATA_IMPORT_DIR + fileElement, 'r', encoding='utf8', errors='ignore')
        # open file to export
        fileExport = codecs.open(RESULT_SAVING_DIR + fileElement, 'w', encoding='utf8', errors='ignore')

        for textIndex, textElement in enumerate(file):
            # shorten text sentences
            pattern = "|".join(sentences_splitter)
            textList = re.split(pattern, textElement)
            # start pseg text
            for subIndex, subElement in enumerate(textList):
                psegResult = []
                total_entity_list = []
                total_label_list = []
                total_dependencies_list = []

                if len(subElement.split(" ")) > 0:
                    for subsubIndex, subsubElement in enumerate(subElement.split(" ")):
                        if len(subsubElement) > 0:
                            psegResult += monpa.pseg(subsubElement)

                    for elementIndex, elementTuple in enumerate(psegResult):
                        # append element into
                        total_entity_list.append(elementTuple[0])
                        total_label_list.append(elementTuple[1])
                        total_dependencies_list.append(np.nan)

                    ''' basic clean up'''
                    relation_list = []
                    relation_index_list = []
                    entity_index_list = []
                    possible_entities = []
                    reformatted_entities = []
                    reformatted_index_list = []

                    for elementIndex, elementTuple in enumerate(psegResult):
                        # delete elements that's between two bracket
                        if psegResult[elementIndex] in bracket_entity_list_first:
                            left_bracket_index = bracket_entity_list_first.index(psegResult[elementIndex][0])
                            findingLimit = (elementIndex + 11) if (elementIndex + 11) <= (
                                len(psegResult)) else len(psegResult)
                            for findingLeftIndex in range(elementIndex + 1, findingLimit):
                                if psegResult[findingLeftIndex] == bracket_entity_list_last[left_bracket_index]:
                                    for removalIndex in range(elementIndex, findingLeftIndex + 1):
                                        total_entity_list[removalIndex] = ""
                                        total_label_list[removalIndex] = ""
                                        total_dependencies_list[removalIndex] = ""
                                    break
                        elif psegResult[elementIndex] in punct_entity_list or \
                                psegResult[elementIndex] in sentences_splitter:
                            # set token that fit certain POS type to ""
                            total_entity_list[elementIndex] = ""
                            total_label_list[elementIndex] = ""
                            total_dependencies_list[elementIndex] = ""

                        # print(total_entity_list, "\n", total_label_list, "\n", total_dependencies_list)

                        ''' get entity pairs and relations '''
                        # get relations
                        if total_dependencies_list[elementIndex] in relation_dependencies_possible_list or \
                                total_label_list[elementIndex] in relation_pos_possible_list or \
                                re.match(relation_pos_re, str(total_label_list[elementIndex]),
                                         flags=re.IGNORECASE) is not None:
                            relation_list.append(total_entity_list[elementIndex])
                            relation_index_list.append(elementIndex)
                        # get entities
                        if total_dependencies_list[elementIndex] in entity_dependencies_possible_list or \
                                total_label_list[elementIndex] in entity_pos_possible_list or \
                                re.match(entity_pos_re, str(total_label_list[elementIndex]),
                                flags=re.IGNORECASE) is not None or total_entity_list[elementIndex] in conjuction_entity_list:
                            entity_index_list.append(elementIndex)
                            possible_entities.append(total_entity_list[elementIndex])

                    ''' refine entities and relations and export '''
                    if len(possible_entities) > 0:
                        combine_entity_name = possible_entities[0]
                        for possibleIndex, possibleElement in enumerate(entity_index_list):

                            isContinuous = False
                            if possibleIndex != 0:
                                if possibleElement == entity_index_list[possibleIndex - 1] + 1:
                                    isContinuous = True
                                    combine_entity_name += str(possible_entities[possibleIndex])
                                else:
                                    isContinuous = False

                                if isContinuous == False:
                                    reformatted_entities.append(combine_entity_name)
                                    reformatted_index_list.append(entity_index_list[possibleIndex - 1])
                                    combine_entity_name = possible_entities[possibleIndex]

                            if possibleIndex == (len(entity_index_list) - 1):
                                reformatted_entities.append(combine_entity_name)
                                reformatted_index_list.append(possibleElement)

                        for elementIndex, elementSingle in enumerate(reformatted_entities):
                            # remove single conjuction char in first or last position
                            if elementSingle[0] in conjuction_entity_list:
                                reformatted_entities[elementIndex] = elementSingle[1:]
                            elif elementSingle[-1] in conjuction_entity_list:
                                reformatted_entities[elementIndex] = elementSingle[:-1]

                    ''' Update entities count dataframe'''
                    total_entities_relation_list = reformatted_entities + relation_list
                    for tokenIndex, tokenElement in enumerate(total_entities_relation_list):
                        if tokenElement in stopword_list or len(tokenElement) == 1 or len(tokenElement) > 8 \
                                or len(tokenElement) == 0:
                            continue

                        if tokenIndex < len(reformatted_entities):
                            if tokenElement not in data_entities.iloc[:, 0].tolist():
                                data_entities = data_entities.append({
                                    "Tokens": str(tokenElement),
                                    "Counts": 1
                                }, ignore_index=True)
                            else:
                                update_index = pd.Index(data_entities.iloc[:, 0].tolist()).get_loc(tokenElement)
                                data_entities.loc[update_index, "Counts"] += 1
                        else:
                            if tokenElement not in data_relations.iloc[:, 0].tolist():
                                data_relations = data_relations.append({
                                    "Tokens": str(tokenElement),
                                    "Counts": 1
                                }, ignore_index=True)
                            else:
                                update_index = pd.Index(data_relations.iloc[:, 0].tolist()).get_loc(tokenElement)
                                data_relations.loc[update_index, "Counts"] += 1

                    if len(data_entities) % 500 == 0:
                        data_entities.to_csv(RESULT_SAVING_DIR + "entity_result.csv", sep=",", mode="w", index=False)
                        data_relations.to_csv(RESULT_SAVING_DIR + "relations_result.csv", sep=",", mode="w", index=False)
                    elif psutil.virtual_memory()[4] < 1000000000:
                        data_entities = data_entities[data_entities["Counts"] > 1]
                        data_relations = data_relations[data_relations["Counts"] > 1]

                    ''' Write out tokenization result '''
                    fileExport.write(str(reformatted_entities) + "," + str(reformatted_index_list) + "," + str(relation_list) + "," + str(relation_index_list))

                fileExport.write("\n")

        file.close()
        fileExport.close()
