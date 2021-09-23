# -*- coding: utf8 -*-
from zhconv import convert
from tqdm import tqdm
import codecs
import matplotlib.pyplot as plt
import networkx as nx
import os
import random
import requests
import sys
import time


''' Configurations '''
MATERIALS_DIR = "C:/Users/User/Desktop/Ricardo/KnowledgeGraph_materials/"
RECONNECT_SECONDS = 1200
RECOGNIZED_EXISTING_WORD_FREQUENCY = 10

''' World Chronology Mandarin config '''
# small dataset
# LOAD_RELATION_PATH = "../../../KnowledgeGraph_materials/results_kg/WorldChronology/SEED_RELATION_WHOLE.csv"
# OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/WorldChronology/STRICT_SEED_RELATION_WHOLE.csv"
# OUTPUT_ENTITY = "../../../KnowledgeGraph_materials/results_kg/WorldChronology/NEW_ENTITY.csv"
# OUTPUT_RELATION = "../../../KnowledgeGraph_materials/results_kg/WorldChronology/NEW_RELATION.csv"
# EXTERNAL_KG_SAVING_PATH = "../dicts/External_KG/CN_Probase.txt"
# OBJECT_DICT_PATH = "../dicts/WorldChronolgy/EntityDict/"
# RELATION_DICT_PATH = "../dicts/WorldChronolgy/RelationDict/"

# large dataset
# LOAD_RELATION_PATH = "../../../KnowledgeGraph_materials/results_kg/WorldChronologyAll/SEED_RELATION_WHOLE.csv"
# OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/WorldChronologyAll/STRICT_SEED_RELATION_WHOLE.csv"
# OUTPUT_ENTITY = "../../../KnowledgeGraph_materials/results_kg/WorldChronologyAll/NEW_ENTITY.csv"
# OUTPUT_RELATION = "../../../KnowledgeGraph_materials/results_kg/WorldChronologyAll/NEW_RELATION.csv"
# EXTERNAL_KG_SAVING_PATH = "../dicts/External_KG/CN_Probase.txt"
# OBJECT_DICT_PATH = "../dicts/WorldChronolgy/EntityDict/"
# RELATION_DICT_PATH = "../dicts/WorldChronolgy/RelationDict/"

''' Semiconductor config '''
# small dataset
LOAD_RELATION_PATH = MATERIALS_DIR + "results_kg/Semiconductor/bootstrapped_relationTriples_Whole.csv"
OUTPUT_PATH_WHOLE = MATERIALS_DIR + "results_kg/Semiconductor/bootstrapped_relationTriples_ExternalKGFiltered_Whole.csv"
OUTPUT_PATH = MATERIALS_DIR + "results_kg/Semiconductor/bootstrapped_relationTriples_ExternalKGFiltered.csv"
EXTERNAL_KG_SAVING_PATH = "../../dicts/External_KG/CN_Probase.txt"
OBJECT_DICT_PATH = "../../dicts/Semiconductor/EntityDict/"
RELATION_DICT_PATH = "../../dicts/Semiconductor/RelationDict/"

# large dataset
# LOAD_RELATION_PATH = "../../../KnowledgeGraph_materials/results_kg/SemiconductorAll/SEED_RELATION_WHOLE.csv"
# OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/SemiconductorAll/STRICT_SEED_RELATION_WHOLE.csv"
# OUTPUT_ENTITY = "../../../KnowledgeGraph_materials/results_kg/SemiconductorAll/NEW_ENTITY.csv"
# OUTPUT_RELATION = "../../../KnowledgeGraph_materials/results_kg/SemiconductorAll/NEW_RELATION.csv"
# EXTERNAL_KG_SAVING_PATH = "../dicts/External_KG/CN_Probase.txt"
# OBJECT_DICT_PATH = "../dicts/Semiconductor/EntityDict/"
# RELATION_DICT_PATH = "../dicts/Semiconductor/RelationDict/"


if __name__ == '__main__':
    ''' Process Starts '''
    data_external_KG = codecs.open(EXTERNAL_KG_SAVING_PATH, mode="r", encoding="utf8", errors="ignore")
    data_import = codecs.open(LOAD_RELATION_PATH, mode="r", encoding="utf8", errors="ignore")
    data_export_whole = codecs.open(OUTPUT_PATH_WHOLE, mode="w", encoding="utf8")
    data_export = codecs.open(OUTPUT_PATH, mode="w", encoding="utf8")

    relation_list = []
    entity_list = []
    new_word_candidate_count_dict = {}
    cn_probase_dict = {}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
        "Accept-Encoding": "*",
        "Connection": "keep-alive"
    }

    # create object list
    for fileIndex, fileElement in enumerate(os.listdir(OBJECT_DICT_PATH)):
        object_dict = codecs.open(OBJECT_DICT_PATH + fileElement, mode="r", encoding="utf8", errors="ignore")
        temp = [line.replace("\r\n", "").replace("\n", "") for line in object_dict.readlines()]
        entity_list += temp

    # create relation list
    for fileIndex, fileElement in enumerate(os.listdir(RELATION_DICT_PATH)):
        relation_dict = codecs.open(RELATION_DICT_PATH + fileElement, mode="r", encoding="utf8", errors="ignore")
        temp = [line.replace("\r\n", "").replace("\n", "") for line in relation_dict.readlines()]
        relation_list += temp

    # create external KG Dict
    for line in data_external_KG.readlines():
        cn_probase_dict[line.split("@")[0]] = line.split("@")[1].replace("[", "").replace("]", "").replace("\n", "")

    # close file, and reopen to add new queries
    data_external_KG.close()
    data_external_KG = codecs.open(EXTERNAL_KG_SAVING_PATH, mode="a", encoding="utf8")

    entity_list = list(set(entity_list))
    relation_list = list(set(relation_list))
    print("Available Entities:", entity_list, len(entity_list), "\nAvailable Relations:",
          relation_list, len(relation_list))

    ''' Dealing with entities and relations those are obvious '''
    lines = data_import.readlines()

    ''' Iteration 1: look up entities in CN External KG'''
    print("Building up dictionary")
    for lineIndex, line in enumerate(tqdm(lines)):
        relation_element_list = line.split("|")[1].split("@")

        # first iteration checking entity and relation that is inside the list
        for relationElementIndex, relationElement in enumerate(relation_element_list):
            ''' query to check if inside CN-Probase '''
            request_retry_count = 0
            if (relationElement not in entity_list) and (relationElement not in relation_list) and (
                    relationElement not in cn_probase_dict.keys()):
                query_word_simplified = convert(relationElement, 'zh-hans')
                retry_count = 0
                while True:
                    try:
                        r = requests.get("http://shuyantech.com/api/cnprobase/ment2ent?q=" + query_word_simplified)

                        result_json = r.json()
                        if result_json["status"] == "ok":
                            cn_probase_dict[relationElement] = result_json["ret"].replace("[", "").replace("]", "")
                            # add new query result into CN_Probase query result file
                            data_external_KG.write(str(relationElement) + "@" + str(result_json["ret"]) + "\n")
                            retry_count = 0
                            break
                        else:
                            # wait 10 seconds and reconnect when can not connect to CN-Probase
                            if retry_count >= 10:
                                print("Retry connection over 10 times, program terminated.")
                                sys.exit(0)
                            else:
                                print("Could not connect to CN-Probase, reconnect again in " + str(
                                    RECONNECT_SECONDS) + " seconds.")
                                time.sleep(RECONNECT_SECONDS)
                                retry_count += 1
                    except requests.exceptions.ConnectionError:
                        request_retry_count += 1
                        print("Retrying connection in", (10 * request_retry_count), "seconds")
                        time.sleep(10 * request_retry_count)
            else:
                if relationElement not in entity_list and relationElement not in relation_list and len(cn_probase_dict[relationElement]) == 0:
                    if relationElement not in new_word_candidate_count_dict.keys():
                        new_word_candidate_count_dict[relationElement] = 0
                    else:
                        new_word_candidate_count_dict[relationElement] += 1


    ''' Iteration II: Export filtered relation triples'''
    print("Filtering Relation Triples...")
    for lineIndex, line in enumerate(tqdm(lines)):
        relation_element_list = line.split("|")[1].split("@")
        IS_IN_DICT = True
        for relationElementIndex, relationElement in enumerate(relation_element_list):
            if (relationElement not in cn_probase_dict.keys() or len(cn_probase_dict[relationElement]) == 0) and (relationElement not in entity_list) and \
                    (relationElement not in relation_list) and \
                    (new_word_candidate_count_dict[relationElement] < RECOGNIZED_EXISTING_WORD_FREQUENCY):
                IS_IN_DICT = False
                break

        if IS_IN_DICT:
            data_export_whole.write(line)
            data_export.write(line.split("|")[0] + "&" + relation_element_list[1] + "\n")

