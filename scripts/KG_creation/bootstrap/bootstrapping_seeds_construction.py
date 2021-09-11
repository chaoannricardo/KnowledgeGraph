# -*- coding: utf8 -*-
"""
Reference:
* https://towardsdatascience.com/how-to-find-shortest-dependency-path-with-spacy-and-stanfordnlp-539d45d28239
* https://stackoverflow.com/questions/32835291/how-to-find-the-shortest-dependency-path-between-two-words-in-python
* https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html

"""
from tqdm import tqdm
import codecs
import networkx as nx
import pandas as pd
import stanza

''' Configurations '''
MATERIALS_DIR = "C:/Users/User/Desktop/Ricardo/KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/"
SEED_RELATION_PATH = MATERIALS_DIR + "duie_train.csv"
SEED_OUTPUT_PATH = MATERIALS_DIR + "seed_relations.csv"
BASIC_SEED_OUTPUT_PATH = MATERIALS_DIR + "seed_relations_basic.csv"
TRIGGER_WORD_OUTPUT_PATH = MATERIALS_DIR + "seed_trigger_word.csv"

if __name__ == '__main__':
    config = {
        'processors': 'tokenize,pos,lemma,depparse',  # Comma-separated list of processors to use
        'lang': 'zh-hant',  # Language code for the language to build the Pipeline in
        'tokenize_model_path': MATERIALS_DIR + 'stanza_resources/zh-hant/tokenize/gsd.pt',
        # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
        'pos_model_path': MATERIALS_DIR + 'stanza_resources/zh-hant/pos/gsd.pt',
        'pos_pretrain_path': MATERIALS_DIR + 'stanza_resources/zh-hant/pretrain/gsd.pt',
        'lemma_model_path': MATERIALS_DIR + 'stanza_resources/zh-hant/lemma/gsd.pt',
        'depparse_model_path': MATERIALS_DIR + 'stanza_resources/zh-hant/depparse/gsd.pt',
        'depparse_pretrain_path': MATERIALS_DIR + 'stanza_resources/zh-hant/pretrain/gsd.pt',
    }

    ''' Process Starts '''
    file_import = codecs.open(SEED_RELATION_PATH, mode="r", encoding="utf8", errors="ignore")
    file_export = codecs.open(SEED_OUTPUT_PATH, mode="w", encoding="utf8")
    file_export_basic = codecs.open(BASIC_SEED_OUTPUT_PATH, mode="w", encoding="utf8")
    file_trigger_word = codecs.open(TRIGGER_WORD_OUTPUT_PATH, mode="w", encoding="utf8")
    nlp = stanza.Pipeline(**config)

    trigger_word_list = []

    for lineIndex, line in enumerate(tqdm(file_import.readlines())):
        if lineIndex == 0:
            continue

        text = line.split("●")[0]
        object_name = line.split("●")[1]
        predicate = line.split("●")[2]
        subject = line.split("●")[3][:-1]

        # create graph to find shortest path
        nodes = []
        tokens = []
        ids = []
        e_1_index = []
        e_1_neglect = []
        e_2_index = []
        e_2_neglect = []
        e_3_index = []
        e_3_neglect = []

        doc = nlp(text)

        data_map = pd.DataFrame({
            "Route": [],
            "NameRoute": [],
            "Edge": []
        })

        for sent in doc.sentences:
            for word in sent.words:
                # print(f"id: {word.id}", f"word: {word.text}", f"head id: {word.head}",
                #       f"head: {sent.words[word.head - 1].text if word.head > 0 else 'root'}", f"deprel: {word.deprel}")

                head = sent.words[word.head - 1].text if word.head > 0 else 'root'
                nodes.append(('{0}'.format(word.id), '{0}'.format(word.head)))
                # append to create map
                data_map = data_map.append({"Route": str(word.id) + "-" + str(word.head),
                                            "NameRoute": str(word.text) + "-" + str(head),
                                            "Edge": str(word.deprel)}, ignore_index=True)
                data_map = data_map.append({"Route": str(word.head) + "-" + str(word.id),
                                            "NameRoute": str(head) + "-" + str(word.text),
                                            "Edge": str(word.deprel)}, ignore_index=True)
                # append to create token list
                tokens.append(word.text)
                ids.append(word.id)

        # construct graph by networkx
        graph = nx.Graph(nodes)

        ''' Create dependency path part '''
        # find vocab
        for tokenIndex, tokenElement in enumerate(tokens):
            for requireIndex, requireElement in enumerate([object_name, predicate, subject]):
                if tokenElement == requireElement:
                    if requireIndex == 0:
                        e_1_index.append(str(tokenIndex + 1))
                        e_1_neglect.append([])
                    elif requireIndex == 1:
                        e_2_index.append(str(tokenIndex + 1))
                        e_2_neglect.append([])
                    else:
                        e_3_index.append(str(tokenIndex + 1))
                        e_3_neglect.append([])
                else:
                    new_token = tokenElement
                    neglect_dependecy_index = []
                    for findingIndex in range(tokenIndex + 1, len(tokens)):
                        new_token += tokens[findingIndex]
                        if new_token == requireElement:
                            for appendingIndex in range(tokenIndex + 1, findingIndex + 1):
                                neglect_dependecy_index.append(str(appendingIndex + 1))
                            # append first token's index
                            if requireIndex == 0:
                                e_1_index.append(str(tokenIndex + 1))
                                e_1_neglect.append(neglect_dependecy_index)
                            elif requireIndex == 1:
                                e_2_index.append(str(tokenIndex + 1))
                                e_2_neglect.append(neglect_dependecy_index)
                            else:
                                e_3_index.append(str(tokenIndex + 1))
                                e_3_neglect.append(neglect_dependecy_index)
                            # append neglect index for further usage
                            break

        # skip if somehow could not find the index
        if len(e_1_index) == 0 or len(e_2_index) == 0 or len(e_3_index) == 0:
            continue

        shortest_path_list = []
        for firstIndex, e_1Element in enumerate(e_1_index):
            for secondIndex, e_2Element in enumerate(e_2_index):
                for thirdIndex, e_3Element in enumerate(e_3_index):
                    edges = []
                    try:
                        predicateIndex = len(nx.shortest_path(graph, source=e_1Element, target=e_2Element))
                        shortest_path_list = nx.shortest_path(graph, source=e_1Element, target=e_2Element) + \
                                             nx.shortest_path(graph, source=e_2Element, target=e_3Element)
                    except nx.exception.NodeNotFound:
                        continue

                    # find edges of the shortest path list
                    edges.append(object_name)
                    for path_index, path_element in enumerate(shortest_path_list):
                        if path_index + 1 == len(shortest_path_list) or\
                                (path_index + 1) in (e_1_neglect[firstIndex] + e_2_neglect[secondIndex] + e_3_neglect[thirdIndex]):
                            continue
                        else:
                            if path_index == predicateIndex:
                                edges.append(predicate)
                            route = path_element + "-" + shortest_path_list[path_index + 1]
                            # print(data_map)
                            # print(route)
                            # print(data_map[data_map["Route"] == route]["Edge"].iloc[0])
                            try:
                                edges.append(data_map[data_map["Route"] == route]["Edge"].iloc[0])
                            except IndexError:
                                pass

                    edges.append(subject)
                    basic_output_list = edges.copy()
                    basic_output_list[0] = "Entity"
                    basic_output_list[-1] = "Entity"
                    # basic_output_list[predicateIndex] = "Predicate"

                    file_export.write("@".join(edges) + "\n")
                    file_export_basic.write("@".join(basic_output_list) + "&" + predicate + "\n")
                    trigger_word_list.append(predicate)

    trigger_word_list = list(set(trigger_word_list))
    file_trigger_word.write(",".join(trigger_word_list))

    # remove duplicate seeds
    file_export = codecs.open(SEED_OUTPUT_PATH, mode="r", encoding="utf8")
    file_export_lines = list(set(file_export.readlines()))
    file_export = codecs.open(SEED_OUTPUT_PATH, mode="w", encoding="utf8")
    file_export.writelines(file_export_lines)

    file_export_basic = codecs.open(BASIC_SEED_OUTPUT_PATH, mode="r", encoding="utf8")
    file_export_basic_lines = list(set(file_export_basic.readlines()))
    file_export_basic = codecs.open(BASIC_SEED_OUTPUT_PATH, mode="w", encoding="utf8")
    file_export_basic.writelines(file_export_basic_lines)