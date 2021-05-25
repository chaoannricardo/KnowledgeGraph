# -*- coding: utf8 -*-
"""
Font Path: (Configuration Reference: https://reurl.cc/Q70DNq)
* Windows: C:/Users/UserName/Anaconda3/Lib/site-packages/matplotlib/mpl-data
* Ubuntu Linux: /home/UserName/anaconda3/lib/python3.6/site-packages/matplotlib/mpl-data/

Some Other Reference:
* https://stackoverflow.com/questions/47094949/labeling-edges-in-networkx

Adding Clauses: font.sans-serif : Microsoft JhengHei, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif
"""
from matplotlib.font_manager import findfont, FontProperties
from tqdm import tqdm
import codecs
import matplotlib.pyplot as plt
import networkx as nx
import os
import random


if __name__ == '__main__':
    ''' Configurations '''
    # LOAD_RELATION_PATH = "../../../KnowledgeGraph_materials/results_kg/WorldChronology/SEED_RELATION_WHOLE.csv"
    LOAD_RELATION_PATH = "../../../KnowledgeGraph_materials/results_kg/WorldChronologyAll/SEED_RELATION_WHOLE.csv"

    OBJECT_DICT_PATH = "../dicts/WorldChronolgy/EntityDict/"
    RELATION_DICT_PATH = "../dicts/WorldChronolgy/RelationDict/"
    NOUN_ENTITY_UPOS = ["PROPN", "NOUN", "PART"]
    RELATIONS_TO_PLOT = 40  # "ALL"
    ITERATION = 10
    plt.rcParams.update({'font.family': 'Microsoft JhengHei'})
    plt.figure(figsize=(50, 50))

    # show font type
    # print(findfont(FontProperties(family=FontProperties().get_family())))

    ''' Process Starts '''
    data_import = codecs.open(LOAD_RELATION_PATH, mode="r", encoding="utf8", errors="ignore")
    G = nx.DiGraph()
    relation_list = []
    element_to_deal_last = []
    entity_list = []
    # variables to construct graph
    graph_trigger_word_dict = {}
    graph_entity_word_list = []
    iteration = 0

    # create object list
    for fileIndex, fileElement in enumerate(os.listdir(OBJECT_DICT_PATH)):
        object_dict = codecs.open(OBJECT_DICT_PATH + fileElement, mode="r", encoding="utf8", errors="ignore")
        temp = [line.replace("\r\n", "").replace("\n", "")  for line in object_dict.readlines()]
        entity_list += temp

    # create relation list
    for fileIndex, fileElement in enumerate(os.listdir(RELATION_DICT_PATH)):
        relation_dict = codecs.open(RELATION_DICT_PATH + fileElement, mode="r", encoding="utf8", errors="ignore")
        temp = [line.replace("\r\n", "").replace("\n", "") for line in relation_dict.readlines()]
        relation_list += temp

    entity_list = list(set(entity_list))
    relation_list = list(set(relation_list))
    print("Available Entities:", entity_list, len(entity_list), "\nAvailable Relations:",
          relation_list, len(relation_list))

    ''' Dealing with entities and relations those are obvious '''
    lines = data_import.readlines()

    random.shuffle(lines)

    relation_triple_constructed_num = 0
    while iteration < ITERATION:
        iteration += 1
        constructed_triples_this_iteration = 0

        print("ITERATION:", iteration, "REMAINING LINES:", len(lines), "\nENTITIY NUMS:",
              len(entity_list), "RELATION NUMS:", len(relation_list),
              "\nCONTRUCTED TUPLES:", len(graph_entity_word_list ))

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
                elif upos_element[relationElementIndex] in ["VERB"]:
                    relation_tokens.append(relationElement)

            # second iteration if possible
            if len(entity_tokens + relation_tokens) == 2:
                for relationElementIndex, relationElement in enumerate(relation_element):
                    if relationElement not in entity_tokens and relationElement not in relation_tokens:
                        if len(entity_tokens) == 1 and len(relation_tokens) == 1:
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
        print("New Added Relation Triples:", [str(graph_entity_word_list[i]) + " " +\
                                              str(graph_trigger_word_dict[graph_entity_word_list[i]]) \
                                              for i in range(relation_triple_constructed_num,
                                                             len(graph_entity_word_list))],
              "\n===================================")
        relation_triple_constructed_num += constructed_triples_this_iteration

    ''' Construct Graph Phase '''
    if RELATIONS_TO_PLOT != "ALL":
        G.add_edges_from(graph_entity_word_list[:RELATIONS_TO_PLOT])
        # pos = nx.spring_layout(G)
        # pos = nx.planar_layout(G)
        pos = nx.circular_layout(G)
        # pos = nx.shell_layout(G)
        # pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx(G, pos, node_size=1000, node_color="y")

        sub_graph_trigger_word_dict = {}
        for tempIndex, tempElement in enumerate(graph_entity_word_list[:RELATIONS_TO_PLOT]):
            sub_graph_trigger_word_dict[tempElement] = graph_trigger_word_dict[tempElement]

        nx.draw_networkx_edge_labels(G, pos, edge_labels=sub_graph_trigger_word_dict)

    else:
        G.add_edges_from(graph_entity_word_list)
        # pos = nx.spring_layout(G)
        # pos = nx.planar_layout(G)
        pos = nx.circular_layout(G)
        # pos = nx.shell_layout(G)
        # pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx(G, pos, node_size=1000, node_color="y")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=graph_trigger_word_dict)

    print("Available Entities:", entity_list, len(entity_list), "\nAvailable Relations:",
          relation_list, len(relation_list))

    plt.show()









