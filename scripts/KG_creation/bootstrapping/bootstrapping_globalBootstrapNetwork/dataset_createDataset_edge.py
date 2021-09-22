# -*- coding: utf8 -*-
from tqdm import tqdm
import codecs
import numpy as np

''' Configurations '''
MATERIALS_DIR = "C:/Users/User/Desktop/Ricardo/KnowledgeGraph_materials/"
SEED_DIR = MATERIALS_DIR + "data_kg/baiduDatasetTranditional_Cleansed/"

# configuration differs on NODE_SET_TYPE
# for single node
OUTPUT_DIR = MATERIALS_DIR + "data_kg/baiduDatasetTranditional_GBN_singleNode_edge/doc_un_1/"
# for whole node set
# OUTPUT_DIR = MATERIALS_DIR + "data_kg/baiduDatasetTranditional_GBN_wholeNode_edge/doc_un_1/"
NODE_SET_TYPE = 0
# 0: single node 1: whole node set

if __name__ == '__main__':
    # open files to write
    file_output_entities = codecs.open(OUTPUT_DIR + "entities.txt", mode="w", encoding="utf8")
    file_output_entities_labels = codecs.open(OUTPUT_DIR + "entity_labels.txt", mode="w", encoding="utf8")
    file_output_labels = codecs.open(OUTPUT_DIR + "labels.txt", mode="w", encoding="utf8")
    file_output_links = codecs.open(OUTPUT_DIR + "links.txt", mode="w", encoding="utf8")
    file_output_pattern_label_vocab = codecs.open(OUTPUT_DIR + "pattern_label_vocab.txt", mode="w", encoding="utf8")
    file_output_pattern_label = codecs.open(OUTPUT_DIR + "pattern_labels.txt", mode="w", encoding="utf8")
    file_output_pattern = codecs.open(OUTPUT_DIR + "patterns.txt", mode="w", encoding="utf8")

    # initiate list to store data
    nodes = []
    node_set_list = []
    patterns = []

    # read in seed file
    file_seed = codecs.open(SEED_DIR + "seed_relations_filtered.csv", encoding="utf8", mode="r", errors="ignore")

    for lineIndex, line in enumerate(tqdm(file_seed.readlines())):
        edge = line.split("&")[-1].replace("\n", "")
        entity_1 = line.split("&")[0].split("@")[0]
        entity_2 = line.split("&")[0].split("@")[-1]
        node_set = entity_1 + "@" + edge + "@" + entity_2

        # store nodes
        for entity in [entity_1, entity_2, edge]:
            if entity not in nodes:
                nodes.append(entity)

        node_set_list.append(node_set)

        try:
            # store patterns
            if edge not in patterns:
                patterns.append(edge)

            # write links: source target weight
            if NODE_SET_TYPE == 0:
                file_output_links.write(
                    "\t".join([str(nodes.index(entity_1)), str(patterns.index(edge)), "1"]) + "\n")
                file_output_links.write(
                    "\t".join([str(nodes.index(entity_2)), str(patterns.index(edge)), "1"]) + "\n")
            elif NODE_SET_TYPE == 1:
                file_output_links.write("\t".join([str(node_set_list.index(node_set)), str(patterns.index(edge)), "1"]) + "\n")

        except ValueError:
            continue

    # write entities & entity_label
    writing_entity_list = nodes if NODE_SET_TYPE == 0 else node_set_list
    for entity in writing_entity_list:
        file_output_entities.write(str(entity) + "\n")
        file_output_entities_labels.write("\t".join([str(writing_entity_list.index(entity)), "0"]) + "\n")

    # write labels
    file_output_labels.write("Entity")

    # write patterns
    for index in range(len(patterns)+1):
        file_output_pattern_label_vocab.write(str(index) + "\n")

    for patternIndex, pattern in enumerate(patterns + patterns):
        file_output_pattern_label.write("\t".join([str(patternIndex), "1"]) + "\n")
        file_output_pattern.write(pattern + "\n")







