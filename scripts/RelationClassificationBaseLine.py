# -*- coding: utf8 -*-
import codecs
import pandas as pd
import json


if __name__ == '__main__':
    DATA_JSON_PATH = "../../KnowledgeGraph_materials/data_kg/baiduDatasetTraditional/RelationExtraction/duie_train.json/duie_train.json"

    ''' Process Starts '''
    data_import = codecs.open(DATA_JSON_PATH, "r", encoding="utf8", errors="ignore")
    data_lines = data_import.readlines()
    for lineIndex, lineElement in enumerate(data_lines):
        line_dict = json.loads(lineElement)
        for keys in line_dict.keys():
            if keys == "text": continue
            linelinedict = json.loads(line_dict[keys])
            print(linelinedict)
        break
