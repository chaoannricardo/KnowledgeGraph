# -*- coding: utf8 -*-
from tqdm import tqdm
import codecs
import json
import pandas as pd


if __name__ == '__main__':
    ''' Configurations '''
    JSON_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTraditional/RelationExtraction/duie_train.json/duie_train.json"
    OUTPUT_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/duie_train.csv"
    REPLACE_CHAR = ["(", "（", "[", "［", "{", "｛", "<", "＜", "〔", "【", "〖", "《", "〈", ")", "）", "]", "］", "}", "｝", ">", "＞", "〕", "】", "〗", "》", "〉"]
    PUNT_CHAR = ["，", "。", "！", "!", "？", "；", ";", "：", "、"]
    NEGLECT_CAHR = ["「", "」", " ", "\n", "-", "——", "?", "－"]
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
                line = line_dict["text"]

                # replace special chars in line
                for charIndex, charElement in enumerate(REPLACE_CHAR):
                    line = line.replace(charElement, "，")

                for charIndex, charElement in enumerate(NEGLECT_CAHR):
                    line = line.replace(charElement, "")

                if line[0] == "，":
                    line = line[1:]

                line_split = [line[i] for i in range(len(line))]
                for charIndex, charElement in enumerate(line_split):
                    if (charIndex+1) == len(line):
                        continue
                    if charElement in PUNT_CHAR and line_split[charIndex + 1] in PUNT_CHAR:
                        line_split[charIndex] = ""

                line = "".join(line_split)
                if line[-1] in PUNT_CHAR:
                    line = line[:-1]

                predicate = relation["predicate"]
                object = relation["object"]["@value"]
                subject = relation["subject"]
                file_export.write("●".join([line, object, predicate, subject]) + "\n")




