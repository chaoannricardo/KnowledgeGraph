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
import os
import pandas as pd
import re
import stanza

if __name__ == '__main__':
    ''' Configurations '''
    # World Chronology Mandarin config
    # BASIC_SEED_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_RELATION_BASIC_FILTER.csv"  # seed dictionary constructed by former script
    # OBJECT_DICT_PATH = "../dicts/WorldChronolgy/EntityDict/"
    # SUBJECT_DICT_PATH = ""  # not using subject dict path for now
    # TRIGGER_WORD_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_TRIGGER_WORD.csv"
    # DATA_IMPORT_PATH = "../../../KnowledgeGraph_materials/data_kg/WorldChronologyMandarin/"  # data used to find new relations
    # NEW_SEED_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210426_result/SEED_RELATION.csv"
    # NEW_BASIC_SEED_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210426_result/SEED_RELATION_BASIC.csv"
    # NEW_WHOLE_SEED_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210426_result/SEED_RELATION_WHOLE.csv"
    # NEW_TRIGGER_WORD_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210426_result/SEED_TRIGGER_WORD.csv"
    # NEW_SEED_ONLY_RELATION = "../../../KnowledgeGraph_materials/results_kg/210426_result/SEED_ONLY_RELATION.csv"

    # semiconductor config
    BASIC_SEED_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_RELATION_BASIC_FILTER.csv"  # seed dictionary constructed by former script
    OBJECT_DICT_PATH = "../dicts/Semiconductor/EntityDict/"
    SUBJECT_DICT_PATH = ""  # not using subject dict path for now
    TRIGGER_WORD_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_TRIGGER_WORD.csv"
    DATA_IMPORT_PATH = "../../../KnowledgeGraph_materials/data_kg/data_normal_wafer_text/"  # data used to find new relations
    NEW_SEED_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210513_result/SEED_RELATION.csv"
    NEW_BASIC_SEED_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210513_result/SEED_RELATION_BASIC.csv"
    NEW_WHOLE_SEED_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210513_result/SEED_RELATION_WHOLE.csv"
    NEW_TRIGGER_WORD_OUTPUT_PATH = "../../../KnowledgeGraph_materials/results_kg/210513_result/SEED_TRIGGER_WORD.csv"
    NEW_SEED_ONLY_RELATION = "../../../KnowledgeGraph_materials/results_kg/210513_result/SEED_ONLY_RELATION.csv"

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
                    "＞", "〕", "】", "〗", "》", "〉", "\r\n", "ˋ"]
    PUNT_CHAR = ["，", "。", "！", "!", "？", "；", ";", "：", "、"]
    CONJUCTION_CHAR = ["的", "之", "及", "與", "等", "前"]
    NEGLECT_CAHR = ["「", "」", " ", "\n", "-", "——", "?"]
    NEGLECT_UPOS = ["PART", "PFA", "NUM"]
    NEGLECT_XPOS = ["SFN"]
    NOUN_ENTITY_UPOS = ["PROPN", "NOUN", "PART"]
    CONTINUE_WORD_UPOS_FIRST = ["PROPN", "ADJ"]  # "NOUN"
    CONTINUE_WORD_UPOS_LAST = ["PART"]
    CONTINUE_WORD_XPOS = []
    CONTINUE_SEARCHING_LIMIT = 2
    TOLERATE_DIFFERENCE = 3
    THIRD_PHASE_COUNT_THERSHOLD = 30
    ITERATIONS = 10

    ''' Process Starts '''
    # data to import
    trigger_word_dict = codecs.open(TRIGGER_WORD_PATH, mode="r", encoding="utf8", errors="ignore")
    data_seed = codecs.open(BASIC_SEED_PATH, mode="r", encoding="utf8", errors="ignore")
    # data to export
    seed_output = codecs.open(NEW_SEED_OUTPUT_PATH, mode="w", encoding="utf8")
    seed_output_basic = codecs.open(NEW_BASIC_SEED_OUTPUT_PATH, mode="w", encoding="utf8")
    seed_output_whole = codecs.open(NEW_WHOLE_SEED_OUTPUT_PATH, mode="w", encoding="utf8")
    seed_output_only_relation = codecs.open(NEW_SEED_ONLY_RELATION, mode="w", encoding="utf8")
    trigger_word_output = codecs.open(NEW_TRIGGER_WORD_OUTPUT_PATH, mode="w", encoding="utf8")

    ''' Construct variables for bootstrapping '''
    seed_relation_list = []
    relation_location_index = []
    object_list = []

    for fileIndex, fileElement in enumerate(os.listdir(OBJECT_DICT_PATH)):
        object_dict = codecs.open(OBJECT_DICT_PATH + fileElement, mode="r", encoding="utf8", errors="ignore")
        temp = [line.replace("\r\n", "").replace("\n", "") for line in object_dict.readlines()]
        object_list += temp

    # set up realtion and relation indexes for bootstrapping
    for lineIndex, line in enumerate(data_seed.readlines()):
        relation_line = line.split("&")[0]
        # relation_type = line.split("&")[1]
        seed_relation_list.append(relation_line)
        relation_location_index.append(relation_line.split("@").index("Predicate"))

    seed_relation_list = list(set(seed_relation_list))  # remove duplicate seed relations
    seed_relation_list = [seed_relation_list[i].split("@") for i in range(len(seed_relation_list))]
    trigger_word_list = trigger_word_dict.readline().split(",")

    print("字典含有詞目：", object_list)

    # load spaCy engines
    nlp = stanza.Pipeline(**config)

    ''' bootstrapping starts '''
    first_phase_relation_list = []
    first_phase_relation_list_whole = []
    first_phase_relation_list_no_trigger = []
    second_phase_relation_list = []
    second_phase_relation_list_whole = []
    second_phase_relation_list_no_trigger = []
    third_phase_relation_list = []
    third_phase_relation_list_whole = []
    third_phase_relation_list_no_trigger = []
    output_upos_whole = []
    output_xpos_whole = []
    trigger_word_candidate = []
    trigger_word_output = []

    for fileIndex, fileElement in enumerate(os.listdir(DATA_IMPORT_PATH)):

        data_import = codecs.open(DATA_IMPORT_PATH + fileElement, mode="r", encoding="utf8", errors="ignore")

        ''' Enumerate over file lines '''
        for lineIndex, line in enumerate(tqdm(data_import.readlines())):

            ''' debugging part '''
            # if lineIndex == 100:
            #     break
            ''' ended '''

            ''' Special line to eliminate years for data '''
            try:
                line = line.split("：")[1]
            except IndexError:
                pass
            ''' Special line finished '''

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

            if len(line) == 0:
                continue

            line = "".join(line_split)
            if line[-1] in PUNT_CHAR:
                line = line[:-1]
            ''' ended '''

            for sublineIndex, subline in enumerate(re.split("|".join(["。", "；"]), line)):
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

                doc = nlp(subline)
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

                        if firstIndex == secondIndex or firstIndex == object_index or object_index == secondIndex or \
                                len(str(tokens[int(firstElement) - 1])) < 2 or len(
                            str(tokens[int(secondElement) - 1])) < 2 or \
                                tokens[int(firstElement) - 1] in PUNT_CHAR or tokens[
                            int(secondElement) - 1] in PUNT_CHAR or \
                                upos[int(firstElement) - 1] in NEGLECT_UPOS or upos[
                            int(secondElement) - 1] in NEGLECT_UPOS or \
                                xpos[int(firstElement) - 1] in NEGLECT_XPOS or xpos[
                            int(secondElement) - 1] in NEGLECT_XPOS:
                            # skip if element duplicate
                            continue
                        else:
                            ''' debugging code '''
                            # print("!!!", tokens[object_index])
                            # print("@@@", tokens[int(ids[object_index]) - 1])
                            ''' ended '''

                            e_1 = str(ids[object_index])
                            e_1_pos = ""
                            e_2 = str(firstElement)
                            e_2_pos = ""
                            e_3 = str(secondElement)
                            e_3_pos = ""

                            ''' 
                            Filter:
                            * check if all entity are inside dictionary, if yes, eliminated
                            * eliminate if conclude more than one verb, or only one noun
                            '''
                            temp_token_count = 0
                            temp_upos_count = 0
                            temp_verb_count = 0

                            for temp_index in [(int(e_1) - 1), (int(e_2) - 1), (int(e_3) - 1)]:
                                # entity in object list check
                                if tokens[temp_index] in object_list:
                                    temp_token_count += 1
                                # upos contain verb check
                                if upos[temp_index] in NOUN_ENTITY_UPOS:
                                    temp_upos_count += 1
                                elif upos[temp_index] in ["VERB"]:
                                    temp_verb_count += 1
                            if temp_token_count == 3 or temp_upos_count < 2 or temp_verb_count > 1:
                                continue
                            elif "CCONJ" in upos and upos.count("CCONJ") == 1:
                                # if str(firstElement) != str(upos.index("CCONJ")) and str(secondElement) != str(
                                #         upos.index("CCONJ")):
                                # print(str(upos.index("CCONJ")))
                                continue
                            ''' Filter Ended '''

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

                            # construct output result
                            for resultIndex, resultElement in enumerate(result_list):

                                # append first token: check if entity is continued by nouns, if so, concat together
                                first_token_index = int(resultElement[0]) - 1
                                left_searching_limit = (first_token_index - CONTINUE_SEARCHING_LIMIT) if \
                                    (first_token_index - CONTINUE_SEARCHING_LIMIT) >= 0 else 0
                                right_searching_limit = (first_token_index + CONTINUE_SEARCHING_LIMIT) if \
                                    (first_token_index + CONTINUE_SEARCHING_LIMIT) < len(tokens) else len(tokens)

                                edges = [tokens[first_token_index]]
                                output_upos = [upos[first_token_index]]
                                output_xpos = [xpos[first_token_index]]

                                # searching left
                                for searchingIndex in range((first_token_index - 1), left_searching_limit, -1):
                                    if upos[first_token_index] not in NOUN_ENTITY_UPOS:
                                        break
                                    elif xpos[searchingIndex] in CONTINUE_WORD_XPOS or upos[
                                        searchingIndex] in CONTINUE_WORD_UPOS_FIRST:
                                        edges[0] = tokens[searchingIndex] + edges[0]

                                # searching right
                                for searchingIndex in range((first_token_index + 1), right_searching_limit):
                                    if upos[first_token_index] not in NOUN_ENTITY_UPOS:
                                        break
                                    elif xpos[searchingIndex] in CONTINUE_WORD_XPOS or upos[
                                        searchingIndex] in CONTINUE_WORD_UPOS_LAST:
                                        edges[0] = edges[0] + tokens[searchingIndex]

                                ''' debugging code '''
                                # print("@@@", tokens[int(resultElement[0]) - 1])
                                ''' ended '''

                                for path_index, path_element in enumerate(resultElement):
                                    if path_index + 1 == len(resultElement):
                                        continue
                                    else:
                                        # appending predicate
                                        if path_index == predicate_index_list[resultIndex]:
                                            predicate_location_in_edge = len(edges)
                                            real_predicate_index = int(path_element) - 1
                                            left_searching_limit = (real_predicate_index - CONTINUE_SEARCHING_LIMIT) if \
                                                (real_predicate_index - CONTINUE_SEARCHING_LIMIT) >= 0 else 0
                                            right_searching_limit = (real_predicate_index + CONTINUE_SEARCHING_LIMIT) if \
                                                (real_predicate_index + CONTINUE_SEARCHING_LIMIT) < \
                                                len(tokens) else len(tokens)

                                            edges.append(tokens[real_predicate_index])
                                            output_upos.append(upos[real_predicate_index])
                                            output_xpos.append(xpos[real_predicate_index])

                                            # searching left
                                            for searchingIndex in range((real_predicate_index - 1),
                                                                        left_searching_limit,
                                                                        -1):
                                                if upos[real_predicate_index] not in NOUN_ENTITY_UPOS or (
                                                        tokens[searchingIndex] in edges[0]):
                                                    break
                                                elif (xpos[searchingIndex] in CONTINUE_WORD_XPOS or upos[
                                                    searchingIndex] in CONTINUE_WORD_UPOS_FIRST):
                                                    edges[predicate_location_in_edge] = tokens[searchingIndex] + edges[
                                                        predicate_location_in_edge]

                                            # searching right
                                            for searchingIndex in range((real_predicate_index + 1),
                                                                        right_searching_limit):
                                                if upos[real_predicate_index] not in NOUN_ENTITY_UPOS or (
                                                        tokens[searchingIndex] in edges[0]):
                                                    break
                                                elif xpos[searchingIndex] in CONTINUE_WORD_XPOS or upos[
                                                    searchingIndex] in CONTINUE_WORD_UPOS_LAST:
                                                    edges[predicate_location_in_edge] = edges[
                                                                                            predicate_location_in_edge] + \
                                                                                        tokens[searchingIndex]

                                        # appending route
                                        route = path_element + "-" + resultElement[path_index + 1]
                                        try:
                                            edges.append(data_map[data_map["Route"] == route]["Edge"].iloc[0])
                                        except IndexError:
                                            pass

                                # appending last entity
                                last_token_index = int(resultElement[-1]) - 1
                                left_searching_limit = (last_token_index - CONTINUE_SEARCHING_LIMIT) if \
                                    (last_token_index - CONTINUE_SEARCHING_LIMIT) >= 0 else 0
                                right_searching_limit = (last_token_index + CONTINUE_SEARCHING_LIMIT) if \
                                    (last_token_index + CONTINUE_SEARCHING_LIMIT) < len(
                                        tokens) else len(tokens)

                                edges.append(tokens[last_token_index])
                                output_upos.append(upos[last_token_index])
                                output_xpos.append(xpos[last_token_index])

                                # searching left
                                for searchingIndex in range((last_token_index - 1), left_searching_limit, -1):
                                    if upos[last_token_index] not in NOUN_ENTITY_UPOS or (
                                            tokens[searchingIndex] in edges[0]) or \
                                            (tokens[searchingIndex] in edges[predicate_location_in_edge]):
                                        break
                                    elif xpos[searchingIndex] in CONTINUE_WORD_XPOS or upos[
                                        searchingIndex] in CONTINUE_WORD_UPOS_FIRST:
                                        edges[-1] = tokens[searchingIndex] + edges[-1]

                                # searching right
                                for searchingIndex in range((last_token_index + 1), right_searching_limit):
                                    if upos[last_token_index] not in NOUN_ENTITY_UPOS or (
                                            tokens[searchingIndex] in edges[0]) or \
                                            (tokens[searchingIndex] in edges[predicate_location_in_edge]):
                                        break
                                    elif (xpos[searchingIndex] in CONTINUE_WORD_XPOS or upos[
                                        searchingIndex] in CONTINUE_WORD_UPOS_FIRST):
                                        edges[-1] = edges[-1] + tokens[searchingIndex]

                                # clean up if including conjuction char
                                if edges[0] not in object_list:
                                    if edges[0][0] in CONJUCTION_CHAR:
                                        edges[0] = edges[0][1:]
                                    if edges[0][-1] in CONJUCTION_CHAR:
                                        edges[0] = edges[0][:-1]

                                if edges[predicate_location_in_edge] not in object_list:
                                    if edges[predicate_location_in_edge][0] in CONJUCTION_CHAR:
                                        edges[predicate_location_in_edge] = edges[predicate_location_in_edge][1:]
                                    if edges[predicate_location_in_edge][-1] in CONJUCTION_CHAR:
                                        edges[predicate_location_in_edge] = edges[predicate_location_in_edge][:-1]

                                if edges[-1] not in object_list:
                                    if edges[-1][0] in CONJUCTION_CHAR:
                                        edges[-1] = edges[-1][1:]
                                    if edges[-1][-1] in CONJUCTION_CHAR:
                                        edges[-1] = edges[-1][:-1]

                                '''
                                Filter:
                                * skip if two entities are same
                                * if dependencies only conclude conj
                                * again eliminate relation triple that is too short
                                '''
                                dependency_list = edges[1:-1].copy()
                                dependency_list[predicate_location_in_edge - 1] = "conj"

                                # if edges[0] == "奧地利":
                                #     print(dependency_list)

                                if len(list({edges[0], edges[predicate_location_in_edge], edges[-1]})) < 3 or \
                                        edges[0] in edges[predicate_location_in_edge] or edges[
                                    predicate_location_in_edge] in edges[0] or \
                                        edges[-1] in edges[predicate_location_in_edge] or edges[
                                    predicate_location_in_edge] in edges[-1] or \
                                        edges[-1] in edges[0] or edges[0] in edges[-1] or \
                                        (len(list(set(dependency_list))) == 1 and "conj" in dependency_list) or \
                                        len(str(edges[0])) < 2 or len(str(edges[-1])) < 2 or \
                                        len(str(edges[predicate_location_in_edge])) < 2:
                                    continue
                                ''' Filter Ended '''

                                # construct basic relation form
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
                                    first_phase_relation_list_no_trigger.append(basic_output_list_no_trigger)

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
                                            second_phase_relation_list_no_trigger.append(basic_output_list_no_trigger)
                                        elif difference <= TOLERATE_DIFFERENCE and is_predicate_same is not True:
                                            third_phase_relation_list.append(basic_output_list)
                                            third_phase_relation_list_whole.append(edges)
                                            third_phase_relation_list_no_trigger.append(basic_output_list_no_trigger)
                                            output_upos_whole.append(output_upos)
                                            output_xpos_whole.append(output_xpos)

                                            # append trigger word candidate
                                            trigger_word_candidate.append(edges[predicate_location_in_edge])

                                ''' debugging code '''
                                # print(len(third_phase_relation_list))
                                # print(len(third_phase_relation_list_whole))
                                ''' ended '''

    ''' Enumerate relations & export '''
    # remove duplicates
    first_phase_relation_list_whole.sort()
    first_phase_relation_list_whole = list(k for k, _ in itertools.groupby(first_phase_relation_list_whole))
    second_phase_relation_list_whole.sort()
    second_phase_relation_list_whole = list(k for k, _ in itertools.groupby(second_phase_relation_list_whole))
    third_phase_relation_list_whole.sort()
    third_phase_relation_list_whole = list(k for k, _ in itertools.groupby(third_phase_relation_list_whole))

    first_phase_relation_list.sort()
    first_phase_relation_list = list(k for k, _ in itertools.groupby(first_phase_relation_list))
    second_phase_relation_list.sort()
    second_phase_relation_list = list(k for k, _ in itertools.groupby(second_phase_relation_list))
    third_phase_relation_list.sort()
    third_phase_relation_list = list(k for k, _ in itertools.groupby(third_phase_relation_list))

    trigger_word_candidate = list(set(trigger_word_candidate))

    # print((first_phase_relation_list_whole))
    # print("++++++++++++++++++++++++++++++++++++++++++++")
    # print((second_phase_relation_list_whole))
    # print("++++++++++++++++++++++++++++++++++++++++++++")
    # print((third_phase_relation_list_whole))

    relation_whole_list = first_phase_relation_list_whole + second_phase_relation_list_whole + third_phase_relation_list_whole
    relation_list = first_phase_relation_list + second_phase_relation_list + third_phase_relation_list

    '''
    Filter: 
    * Calculate count of bootstapping method, eliminate those below thershold
    '''
    data_third_phase_candidate = pd.DataFrame({
        "Candidate": ["@".join(relationElement) for relationElement in third_phase_relation_list_no_trigger]
    })
    data_third_phase_candidate_filter = data_third_phase_candidate.value_counts().reset_index()
    data_third_phase_candidate_filter.to_csv("./temp.csv")
    data_third_phase_candidate_filter = data_third_phase_candidate_filter[data_third_phase_candidate_filter.iloc[:, 1] \
                                                                          > THIRD_PHASE_COUNT_THERSHOLD].iloc[:,
                                        0].tolist()
    ''' Filter Ended '''

    for relationIndex, relationElement in enumerate(relation_list):
        if relationIndex >= len(first_phase_relation_list + second_phase_relation_list):
            if "@".join(relationElement) in data_third_phase_candidate_filter:
                seed_output_basic.write("@".join(relationElement) + "\n")
        else:
            seed_output_basic.write("@".join(relationElement) + "\n")

    for relationIndex, relationElement in enumerate(relation_whole_list):
        # create basic form
        temp_basic_form = relationElement.copy()
        temp_basic_form[0] = temp_basic_form[-1] = "Entity"

        # simple relation constructing
        simple_relation_format = [relationElement[0]]

        for tempIndex, tempElement in enumerate(temp_basic_form):
            if tempElement in trigger_word_candidate:
                temp_basic_form[tempIndex] = "Predicate"
                simple_relation_format.append(relationElement[tempIndex])

        simple_relation_format.append(relationElement[-1])

        # only export if it's beyond threshold
        temp_basic_form = "@".join(temp_basic_form)
        if temp_basic_form in data_third_phase_candidate_filter:
            seed_output_whole.write("@".join(relationElement) + "\n")
            seed_output_only_relation.write("@".join(simple_relation_format) +
                                            "|" + "@".join(output_upos_whole[relationIndex]) +
                                            "|" + "@".join(output_xpos_whole[relationIndex]) + "\n")

        # seed_output_whole.write("@".join(output_upos_whole[relationIndex]) + "\n")
        # seed_output_whole.write("@".join(output_xpos_whole[relationIndex]) + "\n")
