# -*- coding: utf8 -*-
from matplotlib.font_manager import findfont, FontProperties
from tqdm import tqdm
from zhconv import convert
import codecs
import matplotlib.pyplot as plt
import networkx as nx
import os
import random
import requests
import sys
import time

if __name__ == '__main__':
    ''' Configurations '''
    LOAD_RELATION_PATH = "../../../KnowledgeGraph_materials/results_kg/WorldChronology/SEED_RELATION_WHOLE.csv"
    # LOAD_RELATION_PATH = "../../../KnowledgeGraph_materials/results_kg/WorldChronologyAll/SEED_RELATION_WHOLE.csv"

    OBJECT_DICT_PATH = "../dicts/WorldChronolgy/EntityDict/"
    RELATION_DICT_PATH = "../dicts/WorldChronolgy/RelationDict/"
    NOUN_ENTITY_UPOS = ["PROPN", "NOUN", "PART"]
    RECOGNIZED_EXISTING_WORD_FREQUENCY = 10
    ITERATION = 10

    # show font type
    # print(findfont(FontProperties(family=FontProperties().get_family())))

    ''' Process Starts '''
    data_import = codecs.open(LOAD_RELATION_PATH, mode="r", encoding="utf8", errors="ignore")
    G = nx.DiGraph()
    relation_list = []
    element_to_deal_last = []
    entity_list = []
    iteration = 0


    relation_triple_constructed_num = 0
    while iteration < ITERATION:
        iteration += 1
        constructed_triples_this_iteration = 0

        for lineIndex, line in enumerate(lines):
            relation_element = line.split("|")[1].split("@")
            upos_element = line.split("|")[2].split("@")
            xpos_element = line.split("|")[3].split("@")
            entity_tokens = []
            relation_tokens = []
            # first iteration checking entity and relation that is inside the list
            for relationElementIndex, relationElement in enumerate(relation_element):
                if relationElement in entity_list:
                    entity_tokens.append(relationElement)
                elif relationElement in relation_list:
                    relation_tokens.append(relationElement)

            # second loop to find new relation and entity
            if len(entity_tokens + relation_tokens) == 2:
                for relationElementIndex, relationElement in enumerate(relation_element):
                    if (relationElement not in entity_tokens) and (relationElement not in relation_tokens) and \
                            ((relationElement in cn_probase_dict.keys() and (len(cn_probase_dict[relationElement]) > 0)
                              or ((relationElement in new_word_candidate_count_dict.keys() and
                             new_word_candidate_count_dict[relationElement] >= RECOGNIZED_EXISTING_WORD_FREQUENCY)))):
                        if len(entity_tokens) == 1 and len(relation_tokens) == 1 and \
                                upos_element[relationElementIndex] not in ["VERB"]:
                            entity_tokens.append(relationElement)
                        elif len(entity_tokens) == 2:
                            relation_tokens.append(relationElement)

            if len(entity_tokens) == 2 and len(relation_tokens) == 1:
                entity_list += entity_tokens
                relation_list += relation_tokens
                graph_entity_word_list.append((entity_tokens[0], entity_tokens[1]))
                graph_trigger_word_dict[(entity_tokens[0], entity_tokens[1])] = relation_tokens[0]
                lines.remove(line)
                constructed_triples_this_iteration += 1

        # remove duplicates
        entity_list = list(set(entity_list))
        relation_list = list(set(relation_list))
        relation_triple_constructed_num += constructed_triples_this_iteration
