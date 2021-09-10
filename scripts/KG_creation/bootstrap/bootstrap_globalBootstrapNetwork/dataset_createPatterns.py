# -*- coding: utf8 -*-
import codecs
import os

''' Configurations '''
MATERIALS_DIR = "C:/Users/User/Desktop/Ricardo/KnowledgeGraph_materials/"
SEED_DIR = MATERIALS_DIR + "data_kg/baiduDatasetTranditional_Cleansed/"
OUTPUT_DIR = MATERIALS_DIR + "data_kg/baiduDatasetTranditional_GBN/"
# pattern type
# 0: Dependency Pattern
# 1: Relation
PATTERN_TYPE = 0

if __name__ == '__main__':
    file_output_entities = codecs.open(OUTPUT_DIR + "entities.txt", mode="w", encoding="utf8")
    file_output_pattern = codecs.open(OUTPUT_DIR + "patterns.txt", mode="w", encoding="utf8")


    if PATTERN_TYPE == 0:
        # create dependency pattern
        file_seed = codecs.open(SEED_DIR + "seed_relations_basic.csv", encoding="utf8", mode="r", errors="ignore")

        for lineIndex, line in enumerate(file_seed.readlines()):
            edge = line.split("&")[1]
            dependency_path = line.split("&")[0].replace("@", "&")



        pass
    elif PATTERN_TYPE == 1:
        pass
