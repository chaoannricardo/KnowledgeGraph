# -*- coding: utf8 -*-
"""

Reminder:
Remember to remove special line while changing datasource.

"""
from tqdm import tqdm
import codecs
import itertools
import networkx as nx
import re
import spacy

if __name__ == '__main__':
    ''' Configurations '''
    BASIC_SEED_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_RELATION_BASIC.csv"  # seed dictionary constructed by former script
    OBJECT_DICT_PATH = "../../../KnowledgeGraph_materials/data_kg/NationNamesMandarin.txt"
    SUBJECT_DICT_PATH = ""  # not using subject dict path for now
    TRIGGER_WORD_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_TRIGGER_WORD.csv"
    DATA_IMPORT_PATH = "../../../KnowledgeGraph_materials/data_kg/WorldChronologyMandarin.txt"  # data used to find new relations
    NEW_SEED_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210426_result/SEED_RELATION.csv"
    NEW_BASIC_SEED_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210426_result/SEED_RELATION_BASIC.csv"
    NEW_WHOLE_SEED_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210426_result/SEED_RELATION_WHOLE.csv"
    NEW_TRIGGER_WORD_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210426_result/SEED_TRIGGER_WORD.csv"
    SPACY_ENGINE_TYPE = "zh_core_web_trf"  # "zh_core_web_sm" "en_core_web_sm"
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
        relation_type = line.split("&")[1]
        seed_relation_list.append(relation_line)
        relation_location_index.append(relation_line.split("@").index(relation_type.replace("\n", "")))

    seed_relation_list = list(set(seed_relation_list))  # remove duplicate seed relations
    seed_relation_list = [seed_relation_list[i].split("@") for i in range(len(seed_relation_list))]
    trigger_word_list = trigger_word_dict.readline().split(",")
    object_list = [line.replace("\r\n", "") for line in object_dict.readlines()]

    # load spaCy engines
    nlp = spacy.load(SPACY_ENGINE_TYPE)

    ''' bootstrapping starts '''
    first_phase_relation_list = []
    second_phase_relation_list = []
    third_phase_relation_list = []

    for lineIndex, line in enumerate(tqdm([line.replace("\r\n", "") for line in data_import.readlines()])):

        if lineIndex == 100:
            first_phase_relation_list.sort()
            first_phase_relation_list = list(k for k, _ in itertools.groupby(first_phase_relation_list))
            second_phase_relation_list.sort()
            second_phase_relation_list = list(k for k, _ in itertools.groupby(second_phase_relation_list))
            third_phase_relation_list.sort()
            third_phase_relation_list = list(k for k, _ in itertools.groupby(third_phase_relation_list))

            print(first_phase_relation_list)
            print("++++++++++++++++++++++++++++++++++++++++++++")
            print(second_phase_relation_list)
            print("++++++++++++++++++++++++++++++++++++++++++++")
            print(third_phase_relation_list)
            break

        ''' Special line to eliminate years for data '''
        line = line.split("：")[0]
        ''' Special line finished '''

        edges_and_dependencies = []
        edges_list = []
        dependencies_list = []
        pos_list = []

        doc = nlp(line)
        includeObject = False
        for token in doc:
            # check if token list include object list word
            if str(token) in object_list:
                includeObject = True

            for child in token.children:
                # append token to construct graph
                edges_and_dependencies.append(('{0}'.format((token.lower_, token.dep_)),
                                               '{0}'.format((child.lower_, child.dep_))))

            # append data for graph searching
            edges_list.append(str(token))
            dependencies_list.append(token.dep_)
            pos_list.append(token.pos_)

        # skip if line does not conclude object word
        if not includeObject:
            continue

        # get object index
        for objectIndex, objectElement in enumerate(object_list):
            if objectElement in edges_list:
                object_index = edges_list.index(objectElement)
                break

        ''' Construct graph object for further processing '''
        # construct graph by networkx
        graph_test = nx.Graph(edges_and_dependencies)

        # construct graph with all possibilities and do bootstrapping
        for firstIndex, firstElement in enumerate(edges_list):
            for secondIndex, secondElement in enumerate(edges_list):

                predicate_index_list = []
                result_list = []

                if firstIndex == secondIndex or firstIndex == objectIndex or object_index == secondIndex:
                    # skip if element duplicate
                    continue
                else:
                    e_1 = "(\'" + str(edges_list[object_index]) + "\', \'" + dependencies_list[
                        object_index] + "\')"
                    e_1_pos = pos_list[object_index]
                    e_2 = "(\'" + str(edges_list[firstIndex]) + "\', \'" + dependencies_list[firstIndex] + "\')"
                    e_2_pos = pos_list[firstIndex]
                    e_3 = "(\'" + str(edges_list[secondIndex]) + "\', \'" + dependencies_list[secondIndex] + "\')"
                    e_3_pos = pos_list[secondIndex]

                    for constuctionIndexA, graphCandidatesA in enumerate([e_2, e_3]):
                        for constuctionIndexB, graphCandidatesB in enumerate([e_2, e_3]):
                            # skip if candidates are same
                            if graphCandidatesA == graphCandidatesB or \
                                    e_1 == graphCandidatesA or e_1 == graphCandidatesB:
                                continue

                            try:
                                shortest_path_list = nx.shortest_path(graph_test, source=e_1,
                                                                      target=graphCandidatesA) + nx.shortest_path(
                                    graph_test,
                                    source=graphCandidatesA, target=graphCandidatesB)[1:]
                                predicate_index_list.append(
                                    len(nx.shortest_path(graph_test, source=e_1, target=graphCandidatesA)) - 1)
                                result_list.append(shortest_path_list)
                            except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound):
                                pass

                    # construct output result
                    for resultIndex, resultElement in enumerate(result_list):
                        root_included = False
                        output_list = []
                        for elementIndex, element in enumerate(resultElement):
                            # edge name: re.split("\'", element)[1]
                            # dependency type: re.split("\'", element)[3]
                            if elementIndex in [0, predicate_index_list[resultIndex]]:
                                output_list.append(re.split("\'", element)[1])
                                output_list.append(re.split("\'", element)[3])
                            elif elementIndex == (len(resultElement) - 1):
                                output_list.append(re.split("\'", element)[1])
                            else:
                                if re.split("\'", element)[3] == "ROOT":
                                    root_included = True
                                    output_list.append(re.split("\'", resultElement[elementIndex + 1])[3])
                                else:
                                    if root_included:
                                        output_list.append(re.split("\'", resultElement[elementIndex + 1])[3])
                                    else:
                                        output_list.append(re.split("\'", element)[3])

                        basic_output_list = output_list.copy()
                        basic_output_list[0] = "Entity"
                        basic_output_list[-1] = "Entity"

                        # print(basic_output_list)
                        # print(output_list)

                        ''' First Phase - all but entities are same '''
                        if basic_output_list in seed_relation_list:
                            first_phase_relation_list.append(output_list)

                        predicate_index = predicate_index_list[resultIndex]
                        for seedIndex, seedRelation in enumerate(seed_relation_list):
                            whole_list = basic_output_list + seedRelation
                            if len(list(set(whole_list))) == len(seedRelation) and len(seedRelation) == len(basic_output_list):
                                is_predicate_same = False
                                difference = 0
                                for testingIndex, testingElement in enumerate(basic_output_list):
                                    if testingIndex == 0 or testingIndex == (len(basic_output_list) - 1):
                                        continue
                                    else:
                                        if testingIndex != predicate_index and testingElement != seedRelation[testingIndex]:
                                            difference += 1
                                        elif testingIndex == predicate_index and testingElement == seedRelation[testingIndex]:
                                            is_predicate_same = True

                                ''' Second Phase - one of dependencies is different '''
                                ''' Third Phase - trigger word is different '''
                                if difference == 1 and is_predicate_same:
                                    second_phase_relation_list.append(output_list)
                                elif difference == 1 and is_predicate_same is not True:
                                    third_phase_relation_list.append(output_list)
                                else:
                                    # print(difference, is_predicate_same)
                                    pass




































