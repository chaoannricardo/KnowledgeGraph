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
    BASIC_SEED_OUTPUT_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_RELATION_BASIC.csv"
    WHOLE_RESULT_OUTPUT_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_RELATION_WHOLE.csv"
    TRIGGER_WORD_OUTPUT_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_TRIGGER_WORD.csv"
    SPACY_ENGINE_TYPE = "zh_core_web_trf"  # "zh_core_web_sm" "en_core_web_sm"
    ''' Process Starts '''
    file_import = codecs.open(SEED_RELATION_PATH, mode="r", encoding="utf8", errors="ignore")
    file_export = codecs.open(SEED_OUTPUT_PATH, mode="w", encoding="utf8")
    file_export_basic = codecs.open(BASIC_SEED_OUTPUT_PATH, mode="w", encoding="utf8")
    file_export_whole = codecs.open(WHOLE_RESULT_OUTPUT_PATH, mode="w", encoding="utf8")
    file_trigger_word = codecs.open(TRIGGER_WORD_OUTPUT_PATH, mode="w", encoding="utf8")
    nlp = spacy.load(SPACY_ENGINE_TYPE)

    trigger_word_list = []

    for lineIndex, line in enumerate(tqdm(file_import.readlines())):
        if lineIndex == 0:
            continue

        # for debugging
        # if lineIndex == 100:
        #     break

        text = line.split("●")[0]
        object_name = line.split("●")[1]
        predicate = line.split("●")[2]
        subject = line.split("●")[3][:-1]

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

            # append data for graph searching
            edges_list.append(str(token))
            dependencies_list.append(token.dep_)

        # check if object, subject, predicate are inside tokens
        if (object_name not in edges_list) or (subject not in edges_list) or (predicate not in edges_list):
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
        predicate_index_list = []
        if len(object_index) == len(subject_index) == len(predicate_index) == 1:
            try:
                e_1 = "(\'" + str(edges_list[object_index[0]]) + "\', \'" + dependencies_list[object_index[0]] + "\')"
                e_2 = "(\'" + str(edges_list[subject_index[0]]) + "\', \'" + dependencies_list[subject_index[0]] + "\')"
                e_3 = "(\'" + str(edges_list[predicate_index[0]]) + "\', \'" + dependencies_list[
                    predicate_index[0]] + "\')"
                shortest_path_list = nx.shortest_path(graph_test, source=e_1, target=e_3) + nx.shortest_path(graph_test,
                                                                                                             source=e_3,
                                                                                                             target=e_2)[
                                                                                            1:]
                predicate_index_list.append(len(nx.shortest_path(graph_test, source=e_1, target=e_3)) - 1)
                result_list.append(shortest_path_list)
            except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound):
                pass
        else:
            for indexA in object_index:
                for indexB in subject_index:
                    for indexC in predicate_index:
                        try:
                            e_1 = "(\'" + str(edges_list[indexA]) + "\', \'" + dependencies_list[indexA] + "\')"
                            e_2 = "(\'" + str(edges_list[indexB]) + "\', \'" + dependencies_list[indexB] + "\')"
                            e_3 = "(\'" + str(edges_list[indexC]) + "\', \'" + dependencies_list[indexC] + "\')"
                            shortest_path_list = nx.shortest_path(graph_test, source=e_1,
                                                                  target=e_3) + nx.shortest_path(
                                graph_test,
                                source=e_3,
                                target=e_2)[1:]
                            predicate_index_list.append(len(nx.shortest_path(graph_test, source=e_1, target=e_3)) - 1)
                            result_list.append(shortest_path_list)
                        except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound):
                            pass

        # export seed result to file
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

            file_export.write("@".join(output_list) + "\n")
            file_export_basic.write("@".join(basic_output_list) + "&" + predicate + "\n")
            file_export_whole.write("@".join(resultElement) + "\n")
            trigger_word_list.append(predicate)

    trigger_word_list = list(set(trigger_word_list))
    file_trigger_word.write(",".join(trigger_word_list))
