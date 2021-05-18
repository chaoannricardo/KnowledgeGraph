# -*- coding: utf8 -*-
from tqdm import tqdm
import codecs
import matplotlib.pyplot as plt
import networkx as nx


if __name__ == '__main__':
    ''' Configurations '''
    LOAD_RELATION_PATH = "../../../KnowledgeGraph_materials/results_kg/210426_result/SEED_ONLY_RELATION.csv"
    OBJECT_DICT_PATH = "../../../KnowledgeGraph_materials/data_kg/NationNamesMandarin.txt"
    NOUN_ENTITY_UPOS = ["PROPN", "NOUN", "PART"]
    ITERATION = 10

    ''' Process Starts '''
    data_import = codecs.open(LOAD_RELATION_PATH, mode="r", encoding="utf8", errors="ignore")
    object_dict = codecs.open(OBJECT_DICT_PATH, mode="r", encoding="utf8", errors="ignore")
    G = nx.DiGraph()

    object_list = [line.replace("\r\n", "").split(",")[0] for line in object_dict.readlines()]
    element_to_deal_last = []
    trigger_word_list = []
    entity_list = []
    # element to construct graph
    graph_trigger_word_dict = {}
    graph_entity_word_list = []

    ''' Dealing with entities and relations those are obvious '''
    for lineIndex, line in enumerate(tqdm(data_import.readlines())):
        relation_element = line.split("|")[0]
        upos_element = line.split("|")[1]
        xpos_element = line.split("|")[2]
        entity_tokens = []
        relation_tokens = []

        for relationElementIndex, relationElement in enumerate(relation_element):
            if relationElement in object_list:
                entity_tokens.append(relationElement)
            elif upos_element[relationElementIndex] in NOUN_ENTITY_UPOS:
                entity_tokens.append(relationElement)
            elif upos_element[relationElementIndex] in ["VERB"]:
               relation_tokens.append(relationElement)

        if len(entity_tokens) == 2 and len(relation_tokens) == 1:
            entity_list += entity_tokens
            trigger_word_list += relation_tokens[0]
            graph_entity_word_list.append((entity_tokens[0], entity_tokens[1]))
            graph_trigger_word_dict[(entity_tokens[0], entity_tokens[1])] = relation_tokens[0]
        else:
            element_to_deal_last.append(line)

    ''' Considering remaining relations and entities '''
    entity_list = list(set(entity_list))
    trigger_word_list = list(set(trigger_word_list))

    iteration = 0
    while iteration < ITERATION:
        iteration += 1
        print("Dealing with remaining relations, epoch:", iteration)
        print("Remaining relation candidates:", len(element_to_deal_last))

        for lineIndex, line in enumerate(element_to_deal_last):
            relation_element = line.split("|")[0]
            upos_element = line.split("|")[1]
            xpos_element = line.split("|")[2]
            entity_tokens = []
            relation_tokens = []

            for relationElementIndex, relationElement in enumerate(relation_element):
                if relationElement in object_list:
                    entity_tokens.append(relationElement)
                elif upos_element[relationElementIndex] in NOUN_ENTITY_UPOS:
                    entity_tokens.append(relationElement)
                elif upos_element[relationElementIndex] in entity_list:
                    entity_tokens.append(relationElement)
                elif upos_element[relationElementIndex] in ["VERB"]:
                    relation_tokens.append(relationElement)
                elif upos_element[relationElementIndex] in trigger_word_list:
                    relation_tokens.append(relationElement)

            if len(entity_tokens) == 2 and len(relation_tokens) == 1:
                entity_list += entity_tokens
                trigger_word_list += relation_tokens[0]
                graph_entity_word_list.append((entity_tokens[0], entity_tokens[1]))
                graph_trigger_word_dict[(entity_tokens[0], entity_tokens[1])] = relation_tokens[0]
                element_to_deal_last.remove(line)
            else:
                print(line)





