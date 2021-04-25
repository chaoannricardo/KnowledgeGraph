# -*- coding: utf8 -*-
from tqdm import tqdm
import codecs
import json
import pandas as pd


if __name__ == '__main__':
    ''' Configurations '''
    JSON_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTraditional/RelationExtraction/duie_train.json/duie_train.json"
    OUTPUT_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/duie_train.csv"
    ''' Process Starts '''
    file_import = codecs.open(JSON_PATH, mode="r", encoding="utf8", errors="ignore")
    file_export = codecs.open(OUTPUT_PATH, mode="w", encoding="utf8")

    # header
    file_export.write("Text, Object, Predicate, Subject \n")

    line_list = file_import.readlines()

    for lineIndex, lineElement in enumerate(tqdm(line_list)):
        line_dict = json.loads(lineElement)
        spo_list = line_dict["spo_list"]

        for relation in spo_list:
            if relation["predicate"] in line_dict["text"]:
                predicate = relation["predicate"]
                object = relation["object"]["@value"]
                subject = relation["subject"]
                file_export.write("|".join([line_dict["text"], object, predicate, subject]) + "\n")




