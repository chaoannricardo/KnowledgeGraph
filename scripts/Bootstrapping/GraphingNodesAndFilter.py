# -*- coding: utf8 -*-
"""
Font Path: (Configuration Reference: https://reurl.cc/Q70DNq)
* Windows: C:/Users/UserName/Anaconda3/Lib/site-packages/matplotlib/mpl-data
* Ubuntu Linux: /home/UserName/anaconda3/lib/python3.6/site-packages/matplotlib/mpl-data/

Some Other Reference:
* https://stackoverflow.com/questions/47094949/labeling-edges-in-networkx

plt.savefig('Error_SDx.svg', format='svg', bbox_inches="tight")

Adding Clauses: font.sans-serif : Microsoft JhengHei, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif
"""
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
    LOAD_RELATION_PATH = "../../../KnowledgeGraph_materials/results_kg/Semiconductor/SEED_RELATION_WHOLE.csv"
    OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/Semiconductor/STRICT_SEED_RELATION_WHOLE.csv"
    OUTPUT_ENTITY = "../../../KnowledgeGraph_materials/results_kg/Semiconductor/NEW_ENTITY.csv"
    OUTPUT_RELATION = "../../../KnowledgeGraph_materials/results_kg/Semiconductor/NEW_RELATION.csv"
    EXTERNAL_KG_SAVING_PATH = "../dicts/External_KG/CN_Probase.txt"
    OBJECT_DICT_PATH = "../dicts/Semiconductor/EntityDict/"
    RELATION_DICT_PATH = "../dicts/Semiconductor/RelationDict/"

    # large dataset
    # LOAD_RELATION_PATH = "../../../KnowledgeGraph_materials/results_kg/SemiconductorAll/SEED_RELATION_WHOLE.csv"
    # OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/SemiconductorAll/STRICT_SEED_RELATION_WHOLE.csv"
    # OUTPUT_ENTITY = "../../../KnowledgeGraph_materials/results_kg/SemiconductorAll/NEW_ENTITY.csv"
    # OUTPUT_RELATION = "../../../KnowledgeGraph_materials/results_kg/SemiconductorAll/NEW_RELATION.csv"
    # EXTERNAL_KG_SAVING_PATH = "../dicts/External_KG/CN_Probase.txt"
    # OBJECT_DICT_PATH = "../dicts/SemiconductorAll/EntityDict/"
    # RELATION_DICT_PATH = "../dicts/SemiconductorAll/RelationDict/"

    NOUN_ENTITY_UPOS = ["PROPN", "NOUN", "PART"]
    RELATIONS_TO_PLOT = 40  # "ALL"
    RECOGNIZED_EXISTING_WORD_FREQUENCY = 10
    RECONNECT_SECONDS = 1200
    ITERATION = 10
    plt.rcParams.update({'font.family': 'Microsoft JhengHei'})
    plt.figure(figsize=(50, 50))

    # show font type
    # print(findfont(FontProperties(family=FontProperties().get_family())))

    ''' Process Starts '''
    data_external_KG = codecs.open(EXTERNAL_KG_SAVING_PATH, mode="r", encoding="utf8", errors="ignore")
    data_import = codecs.open(LOAD_RELATION_PATH, mode="r", encoding="utf8", errors="ignore")
    data_export = codecs.open(OUTPUT_PATH, mode="w", encoding="utf8")
    data_entity = codecs.open(OUTPUT_ENTITY, mode="w", encoding="utf8")
    data_relation = codecs.open(OUTPUT_RELATION, mode="w", encoding="utf8")
    G = nx.DiGraph()
    relation_list = []
    element_to_deal_last = []
    entity_list = []
    new_word_candidate_count_dict = {}
    cn_probase_dict = {}
    iteration = 0
    # variables to construct graph
    graph_trigger_word_dict = {}
    graph_entity_word_list = []

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
        cn_probase_dict[line.split("@")[0]] = line.split("@")[1]

    # close file, and reopen to add new queries
    data_external_KG.close()
    data_external_KG = codecs.open(EXTERNAL_KG_SAVING_PATH, mode="a", encoding="utf8")

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
              "\nCONTRUCTED TUPLES:", len(graph_entity_word_list))

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
                else:
                    ''' query to check if inside CN-Probase '''
                    # query unseen word in first iteration
                    if iteration == 1:
                        if relationElement not in cn_probase_dict.keys():
                            query_word_simplified = convert(relationElement, 'zh-hans')
                            retry_count = 0
                            while True:
                                r = requests.get(
                                    "http://shuyantech.com/api/cnprobase/ment2ent?q=" + query_word_simplified,
                                    verify=False)
                                result_json = r.json()
                                if result_json["status"] == "ok":
                                    cn_probase_dict[relationElement] = result_json["ret"]
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
                                        print("Could not connect to CN-Probase, reconnect again in " + str(RECONNECT_SECONDS) + " seconds.")
                                        time.sleep(RECONNECT_SECONDS)
                                        retry_count += 1
                        else:
                            if len(cn_probase_dict[relationElement]) == 0:
                                if relationElement not in new_word_candidate_count_dict.keys():
                                    new_word_candidate_count_dict[relationElement] = 0
                                else:
                                    new_word_candidate_count_dict[relationElement] += 1

                    try:
                        if len(cn_probase_dict[relationElement]) > 0 or \
                                (relationElement in new_word_candidate_count_dict.keys() and
                                 new_word_candidate_count_dict[relationElement] >= RECOGNIZED_EXISTING_WORD_FREQUENCY):
                            if upos_element[relationElementIndex] in ["VERB"]:
                                relation_tokens.append(relationElement)
                                print(relationElement)
                    except KeyError:
                        pass

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
        print("New Added Relation Triples:", [str(graph_entity_word_list[i]) + " " + \
                                              str(graph_trigger_word_dict[graph_entity_word_list[i]]) \
                                              for i in range(relation_triple_constructed_num,
                                                             len(graph_entity_word_list))],
              "\n===================================")
        relation_triple_constructed_num += constructed_triples_this_iteration

    ''' Export Phase '''
    for key in graph_trigger_word_dict.keys():
        data_export.write(key[0] + "@" + graph_trigger_word_dict[key] + "@" + key[1] + "\n")

    for entity in entity_list:
        data_entity.write(entity + "\n")

    for relation in relation_list:
        data_relation.write(relation + "\n")

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
