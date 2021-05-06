# -*- coding: utf8 -*-
"""

Reminder:
Remember to remove special line while changing datasource.

Reference:
* https://stackoverflow.com/questions/57078311/load-stanfordnlp-model-locally

"""
from tqdm import tqdm
import codecs
import itertools
import networkx as nx
import pandas as pd
import re
import stanza


if __name__ == '__main__':
    ''' Configurations '''
    BASIC_SEED_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_RELATION_BASIC_FILTER.csv"  # seed dictionary constructed by former script
    OBJECT_DICT_PATH = "../../../KnowledgeGraph_materials/data_kg/NationNamesMandarin.txt"
    SUBJECT_DICT_PATH = ""  # not using subject dict path for now
    TRIGGER_WORD_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_TRIGGER_WORD.csv"
    DATA_IMPORT_PATH = "../../../KnowledgeGraph_materials/data_kg/WorldChronologyMandarin.txt"  # data used to find new relations
    NEW_SEED_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210426_result/SEED_RELATION.csv"
    NEW_BASIC_SEED_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210426_result/SEED_RELATION_BASIC.csv"
    NEW_WHOLE_SEED_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210426_result/SEED_RELATION_WHOLE.csv"
    NEW_TRIGGER_WORD_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210426_result/SEED_TRIGGER_WORD.csv"
    config = {
        'processors': 'tokenize,pos,lemma,depparse',  # Comma-separated list of processors to use
        'lang': 'zh-hant',  # Language code for the language to build the Pipeline in
        'tokenize_model_path': '../../../KnowledgeGraph_materials/stanza_resources/zh-hant/tokenize/gsd.pt',
        # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
        'pos_model_path': '../../../KnowledgeGraph_materials/stanza_resources/zh-hant/pos/gsd.pt',
        'pos_pretrain_path': '../../../KnowledgeGraph_materials/stanza_resources/zh-hant/pretrain/gsd.pt',
        'lemma_model_path': '../../../KnowledgeGraph_materials/stanza_resources/zh-hant/lemma/gsd.pt',
        'depparse_model_path': '../../../KnowledgeGraph_materials/stanza_resources/zh-hant/depparse/gsd.pt',
        'depparse_pretrain_path': '../../../KnowledgeGraph_materials/stanza_resources/zh-hant/pretrain/gsd.pt',
    }
    REPLACE_CHAR = ["(", "（", "[", "［", "{", "｛", "<", "＜", "〔", "【", "〖", "《", "〈", ")", "）", "]", "］", "}", "｝", ">",
                    "＞", "〕", "】", "〗", "》", "〉", "。"]
    PUNT_CHAR = ["，", "。", "！", "!", "？", "；", ";", "：", "、"]
    NEGLECT_CAHR = ["「", "」", " ", "\n", "-", "——", "?"]
    NEGLECT_UPOS = ["PART", "PFA", "NUM"]
    NEGLECT_XPOS = ["SFN"]
    TOLERATE_DIFFERENCE = 3
    ITERATIONS = 10

    ''' Process Starts '''
    # data to import
    data_import = codecs.open(DATA_IMPORT_PATH, mode="r", encoding="utf8", errors="ignore")
    object_dict = codecs.open(OBJECT_DICT_PATH, mode="r", encoding="utf8", errors="ignore")
    trigger_word_dict = codecs.open(TRIGGER_WORD_PATH, mode="r", encoding="utf8", errors="ignore")
    data_seed = codecs.open(BASIC_SEED_PATH, mode="r", encoding="utf8", errors="ignore")
    # data to export
    seed_output = codecs.open(NEW_SEED_OUTPUT_PATH, mode="w", encoding="utf8")
    seed_output_basic = codecs.open(NEW_BASIC_SEED_OUTPUT_PATH, mode="w", encoding="utf8")
    seed_output_whole = codecs.open(NEW_WHOLE_SEED_OUTPUT_PATH, mode="w", encoding="utf8")
    trigger_word_output = codecs.open(NEW_TRIGGER_WORD_OUTPUT_PATH, mode="w", encoding="utf8")

    # process formally starts
    ''' Construct variables for bootstrapping '''
    seed_relation_list = []
    relation_location_index = []

    # set up realtion and relation indexes for bootstrapping
    for lineIndex, line in enumerate(data_seed.readlines()):
        relation_line = line.split("&")[0]
        # relation_type = line.split("&")[1]
        seed_relation_list.append(relation_line)
        relation_location_index.append(relation_line.split("@").index("Predicate"))

    seed_relation_list = list(set(seed_relation_list))  # remove duplicate seed relations
    seed_relation_list = [seed_relation_list[i].split("@") for i in range(len(seed_relation_list))]
    trigger_word_list = trigger_word_dict.readline().split(",")
    object_list = [line.replace("\r\n", "") for line in object_dict.readlines()]

    # load spaCy engines
    nlp = stanza.Pipeline(**config)

    ''' bootstrapping starts '''
    first_phase_relation_list = []
    first_phase_relation_list_whole = []
    second_phase_relation_list = []
    second_phase_relation_list_whole = []
    third_phase_relation_list = []
    third_phase_relation_list_whole = []

    for lineIndex, line in enumerate(tqdm([line.replace("\r\n", "") for line in data_import.readlines()])):

        ''' debugging code '''
        if lineIndex == 100:
        # if lineIndex == len([line.replace("\r\n", "") for line in data_import.readlines()]) - 1:
            first_phase_relation_list_whole.sort()
            first_phase_relation_list_whole = list(k for k, _ in itertools.groupby(first_phase_relation_list_whole))
            second_phase_relation_list_whole.sort()
            second_phase_relation_list_whole = list(k for k, _ in itertools.groupby(second_phase_relation_list_whole))
            third_phase_relation_list_whole.sort()
            third_phase_relation_list_whole = list(k for k, _ in itertools.groupby(third_phase_relation_list_whole))

            print((first_phase_relation_list_whole))
            print("++++++++++++++++++++++++++++++++++++++++++++")
            print((second_phase_relation_list_whole))
            print("++++++++++++++++++++++++++++++++++++++++++++")
            print((third_phase_relation_list_whole))
            break
        ''' debigging code finished '''

        ''' Special line to eliminate years for data '''
        try:
            line = line.split("：")[1]
        except IndexError:
            pass
        ''' Special line finished '''

        # create graph to find shortest path
        nodes = []
        tokens = []
        ids = []
        upos = []
        xpos = []
        e_1_index = []
        e_1_neglect = []
        e_2_index = []
        e_2_neglect = []
        e_3_index = []
        e_3_neglect = []
        pos_list = []

        ''' Preprocessing of text line '''
        # replace special chars in line
        for charIndex, charElement in enumerate(REPLACE_CHAR):
            line = line.replace(charElement, "，")

        for charIndex, charElement in enumerate(NEGLECT_CAHR):
            line = line.replace(charElement, "")

        if line[0] == "，":
            line = line[1:]

        line_split = [line[i] for i in range(len(line))]
        for charIndex, charElement in enumerate(line_split):
            if (charIndex + 1) == len(line):
                continue
            if charElement in PUNT_CHAR and line_split[charIndex + 1] in PUNT_CHAR:
                line_split[charIndex] = ""

        line = "".join(line_split)
        if line[-1] in PUNT_CHAR:
            line = line[:-1]

        ''' ended '''

        doc = nlp(line)
        includeObject = False

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
                upos.append(word.upos)
                xpos.append(word.xpos)

                # check if token list include object list word
                if str(word.text) in object_list:
                    includeObject = True

        # skip if line does not include object word
        if not includeObject:
            continue

        # get object index
        object_index = 99999
        for objectIndex, objectElement in enumerate(object_list):
            if objectElement in tokens:
                object_index = tokens.index(objectElement)
                print(tokens[object_index])
                break

        if object_index == 99999:
            continue

        ''' Construct graph object for further processing '''
        # construct graph by networkx
        graph_test = nx.Graph(nodes)

        # construct graph with all possibilities and do bootstrapping
        for firstIndex, firstElement in enumerate(ids):
            for secondIndex, secondElement in enumerate(ids):

                predicate_index_list = []
                result_list = []

                if firstIndex == secondIndex or firstIndex == object_index or object_index == secondIndex or\
                        len(str(tokens[int(firstElement) - 1])) < 2 or len(str(tokens[int(secondElement) - 1])) < 2 or\
                        tokens[int(firstElement) - 1] in PUNT_CHAR or tokens[int(secondElement) - 1] in PUNT_CHAR or\
                        upos[int(firstElement) - 1] in NEGLECT_UPOS or upos[int(secondElement) - 1] in NEGLECT_UPOS or\
                        xpos[int(firstElement) - 1] in NEGLECT_XPOS or xpos[int(secondElement) - 1] in NEGLECT_XPOS:
                    # skip if element duplicate
                    print("hi")
                    continue
                else:
                    e_1 = str(ids[object_index])
                    print("!!!", tokens[object_index])
                    print("@@@", tokens[int(ids[object_index]) - 1])
                    e_1_pos = ""
                    e_2 = str(firstElement)
                    e_2_pos = ""
                    e_3 = str(secondElement)
                    e_3_pos = ""

                    for constuctionIndexA, graphCandidatesA in enumerate([e_2, e_3]):
                        for constuctionIndexB, graphCandidatesB in enumerate([e_2, e_3]):
                            # skip if candidates are same
                            if graphCandidatesA == graphCandidatesB or \
                                    e_1 == graphCandidatesA or e_1 == graphCandidatesB:
                                continue

                            shortest_path_list = nx.shortest_path(graph_test, source=e_1,
                                                                  target=graphCandidatesA) + nx.shortest_path(
                                graph_test,
                                source=graphCandidatesA, target=graphCandidatesB)[1:]
                            predicate_index_list.append(
                                len(nx.shortest_path(graph_test, source=e_1, target=graphCandidatesA)) - 1)
                            result_list.append(shortest_path_list)

                            try:
                                pass
                            except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound):
                                pass

                    # construct output result
                    for resultIndex, resultElement in enumerate(result_list):
                        edges = [tokens[int(resultElement[0]) - 1]]
                        print("@@@", tokens[int(resultElement[0]) - 1])
                        for path_index, path_element in enumerate(resultElement):
                            if path_index + 1 == len(resultElement):
                                continue
                            else:
                                if path_index == predicate_index_list[resultIndex]:
                                    edges.append(tokens[int(path_element) - 1])
                                route = path_element + "-" + resultElement[path_index + 1]
                                try:
                                    edges.append(data_map[data_map["Route"] == route]["Edge"].iloc[0])
                                except IndexError:
                                    pass

                        edges.append(tokens[int(resultElement[-1]) - 1])

                        basic_output_list = edges.copy()
                        basic_output_list_no_trigger = edges.copy()
                        basic_output_list[0] = "Entity"
                        basic_output_list[-1] = "Entity"
                        basic_output_list_no_trigger[0] = "Entity"
                        basic_output_list_no_trigger[-1] = "Entity"
                        basic_output_list_no_trigger[predicate_index_list[resultIndex] + 1] = "Predicate"

                        # print(edges)
                        # print(basic_output_list)

                        ''' First Phase - all but entities are same '''
                        if basic_output_list in seed_relation_list:
                            first_phase_relation_list.append(basic_output_list)
                            first_phase_relation_list_whole.append(edges)

                        predicate_index = (predicate_index_list[resultIndex] + 1)

                        for seedIndex, seedRelation in enumerate(seed_relation_list):
                            if len(seedRelation) == len(basic_output_list):
                                is_predicate_same = False
                                difference = 0
                                for testingIndex, testingElement in enumerate(basic_output_list):
                                    if testingIndex == 0 or testingIndex == (len(basic_output_list) - 1):
                                        continue
                                    else:
                                        if testingElement == seedRelation[testingIndex]:
                                            if testingIndex == predicate_index:
                                                is_predicate_same = True
                                        else:
                                            difference += 1

                                ''' Second Phase - one of dependencies is different '''
                                ''' Third Phase - trigger word is different '''
                                if difference <= TOLERATE_DIFFERENCE and is_predicate_same:
                                    second_phase_relation_list.append(basic_output_list)
                                    second_phase_relation_list_whole.append(edges)
                                elif difference <= TOLERATE_DIFFERENCE and is_predicate_same is not True:
                                    third_phase_relation_list.append(basic_output_list)
                                    third_phase_relation_list_whole.append(edges)
                                else:
                                    # print(difference, is_predicate_same)
                                    pass
                        # print(len(third_phase_relation_list))
                        # print(len(third_phase_relation_list_whole))

    relation_whole_list = first_phase_relation_list_whole + second_phase_relation_list_whole + third_phase_relation_list_whole
    relation_list = first_phase_relation_list + second_phase_relation_list + third_phase_relation_list

    for relationIndex, relationElement in enumerate(relation_list):
        seed_output_basic.write("@".join(relationElement) + "\n")

    for relationIndex, relationElement in enumerate(relation_whole_list):
        seed_output_whole.write("@".join(relationElement) + "\n")




































