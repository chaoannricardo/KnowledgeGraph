# -*- coding: utf8 -*-
"""
Reference:
* https://towardsdatascience.com/how-to-find-shortest-dependency-path-with-spacy-and-stanfordnlp-539d45d28239

"""
from nltk import Tree
from tqdm import tqdm
import codecs
import networkx as nx
import re
import spacy


def find_element_nested_list(element, list_source, index_list, index="start"):
    # find element recursively
    for the_index, subList in enumerate(list_source):
        if type(subList) is not list:
            if list_source[the_index] == element:
                if index != "start": index_list.append(str(index))
                index_list.append(str(the_index))
                index_list.append("@")
        else:
            find_element_nested_list(element, subList, index_list, str(the_index))

    index_result = "-".join(index_list)
    index_result = re.split("|".join(["-@-", "-@"]), index_result)[:-1]

    return index_result


if __name__ == '__main__':
    ''' Configurations '''
    SEED_RELATION_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/duie_train.csv"
    SEED_OUTPUT_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_RELATION.csv"
    WHOLE_RESULT_OUTPUT_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_RELATION_WHOLE.csv"
    TRIGGER_WORD_OUTPUT_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_TRIGGER_WORD.csv"
    SPACY_ENGINE_TYPE = "zh_core_web_trf"  # "zh_core_web_sm" "en_core_web_sm"
    ''' Process Starts '''
    file_import = codecs.open(SEED_RELATION_PATH, mode="r", encoding="utf8", errors="ignore")
    file_export = codecs.open(SEED_OUTPUT_PATH, mode="w", encoding="utf8")
    file_export_whole = codecs.open(WHOLE_RESULT_OUTPUT_PATH, mode="w", encoding="utf8")
    file_trigger_word = codecs.open(TRIGGER_WORD_OUTPUT_PATH, mode="w", encoding="utf8")
    nlp = spacy.load(SPACY_ENGINE_TYPE)

    for lineIndex, line in enumerate(tqdm(file_import.readlines())):
        if lineIndex == 0:
            continue

        text = line.split("|")[0]
        object_name = line.split("|")[1]
        predicate = line.split("|")[2]
        subject = line.split("|")[3]

        # Load spacy's dependency tree into a networkx graph
        edges = []
        edges_and_dependencies = []
        edges_list = []
        dependencies_list = []
        object_index = []
        subject_index = []
        predicate_index = []
        result_list = []

        doc = nlp(text)

        for token in doc:
            for child in token.children:
                # append token to construct graph
                edges.append(('{0}'.format(token.lower_),
                              '{0}'.format(child.lower_)))
                edges_and_dependencies.append(('{0}'.format((token.lower_, token.dep_)),
                                               '{0}'.format((child.lower_, child.dep_))))

            # append data for graph searchin
            edges_list.append(str(token))
            dependencies_list.append(token.dep_)

        # check if object, subject, predicate are inside tokens
        if object_name not in edges_list or subject not in edges_list or predicate not in edges_list:
            continue

        # construct graph by networkx
        graph = nx.Graph(edges)
        graph_test = nx.Graph(edges_and_dependencies)

        # print(edges_list)
        # print(dependencies_list)

        for index, element in enumerate(edges_list):
            if str(element) == object_name:
                object_index.append(index)
            elif str(element) == subject:
                subject_index.append(index)
            elif str(element) == predicate:
                predicate_index.append(index)

        shortest_path_list = []  # initiate object to prevent warnings
        if len(object_index) == len(subject_index) == len(predicate_index) == 1:
            e_1 = "(\'" + str(edges_list[object_index[0]]) + "\', \'" + dependencies_list[object_index[0]] + "\')"
            e_2 = "(\'" + str(edges_list[subject_index[0]]) + "\', \'" + dependencies_list[subject_index[0]] + "\')"
            e_3 = "(\'" + str(edges_list[predicate_index[0]]) + "\', \'" + dependencies_list[predicate_index[0]] + "\')"
            shortest_path_list = nx.shortest_path(graph_test, source=e_1, target=e_3) + nx.shortest_path(graph_test,
                                                                                                         source=e_3,
                                                                                                         target=e_2)[1:]
            result_list.append(shortest_path_list)
        else:
            for indexA in object_index:
                for indexB in subject_index:
                    for indexC in predicate_index:
                        e_1 = "(\'" + str(edges_list[indexA]) + "\', \'" + dependencies_list[indexA] + "\')"
                        e_2 = "(\'" + str(edges_list[indexB]) + "\', \'" + dependencies_list[indexB] + "\')"
                        e_3 = "(\'" + str(edges_list[indexC]) + "\', \'" + dependencies_list[indexC] + "\')"
                        shortest_path_list = nx.shortest_path(graph_test, source=e_1, target=e_2)[1:]
                        result_list.append(shortest_path_list)

        # export seed result to file
        for resultIndex, resultElement in enumerate(result_list):
            root_included = False
            output_list = []
            for elementIndex, element in enumerate(shortest_path_list):
                if elementIndex not in [0, len(shortest_path_list) - 1]:
                    output_list.append(re.split("\'", element)[3])
                else:
                    if root_included:
                        output_list.append(re.split("\'", element)[3])
                    output_list.append(re.split("\'", element)[1])

            file_export.write("|".join(output_list) + "\n")
            file_export_whole.write("+".join(resultElement) + "\n")
            file_trigger_word.write(predicate + ",")
