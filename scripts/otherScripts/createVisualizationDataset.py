# -*- coding: utf8 -*-
import codecs
import os


if __name__ == '__main__':
    ''' Configurations '''
    RelationTriplePath = "../../../KnowledgeGraph_materials/results_kg/WorldChronology/SEED_RELATION_WHOLE.csv"
    DATA_EXPORT_PATH = "../../visualization/example _WorldChronology/datasets/WorldChronology.json"
    # RelationTriplePath = "../../../KnowledgeGraph_materials/results_kg/WorldChronologyAll/SEED_RELATION_WHOLE.csv"
    # DATA_EXPORT_PATH = "../../visualization/example_WorldChronologyAll/datasets/WorldChronologyAll.json"

    ''' Process Starts '''
    relation_triple_list = []
    entity_count_dict = {}
    entity_group_dict = {}
    relation_triple_text_dict = {}
    entity_with_single_occurence_searching_dict = {}
    data_import = codecs.open(RelationTriplePath, mode="r", encoding="utf8", errors="ignore")
    data_export = codecs.open(DATA_EXPORT_PATH, mode="w", encoding="utf8")

    # summarize with counts of each entity
    for line in data_import.readlines():
        relation_triple = line.split("|")[1].split("@")
        relation_triple_list.append(relation_triple)

        # adding count of occurence of entity
        for entity in [relation_triple[0], relation_triple[2]]:
            if entity in entity_count_dict.keys():
                entity_count_dict[entity] += 1
            else:
                entity_count_dict[entity] = 1
                entity_with_single_occurence_searching_dict[entity] = relation_triple[0] if \
                    entity == relation_triple[2] else relation_triple[2]

        # adding text path of each relation triple
        if (relation_triple[0], relation_triple[2]) in relation_triple_text_dict.keys() and\
            relation_triple[1] not in relation_triple_text_dict[(relation_triple[0], relation_triple[2])].split("&"):
            relation_triple_text_dict[(relation_triple[0], relation_triple[2])] += ("&" + relation_triple[1])
            print("Multiple Relation within two entities!")
        else:
            relation_triple_text_dict[(relation_triple[0], relation_triple[2])] = relation_triple[1]

    # create group
    group = 1
    temp_list = []
    for relation_triple in relation_triple_list:
        for entity in [relation_triple[0], relation_triple[2]]:
            if entity not in entity_group_dict.keys():
                if entity_count_dict[entity] > 1:
                    entity_group_dict[entity] = group
                    group += 1
                elif entity_count_dict[entity] == 1:
                    temp_list.append(entity)

    temp_list = list(set(temp_list))

    for entity in temp_list:
        try:
            entity_group_dict[entity] = entity_group_dict[entity_with_single_occurence_searching_dict[entity]]
        except KeyError:
            # for those both entities only occur once
            entity_group_dict[entity] = group
            group += 1

    # export to graph json format
    data_export.write("{\n\"nodes\":[\n")
    for entityIndex, entity in enumerate(entity_group_dict.keys()):
        if entityIndex != len(entity_group_dict.keys()) - 1:
            data_export.write("{\"id\":\"" + entity + "\", \"group\": " + str(entity_group_dict[entity]) + "},\n")
        else:
            data_export.write("{\"id\":\"" + entity + "\", \"group\": " + str(entity_group_dict[entity]) + "}\n],\n\"links\":[\n")

    for relation_triple_index, relation_triple in enumerate(relation_triple_list):
        if relation_triple_index != len(relation_triple_list) - 1:
            data_export.write("{\"source\": \"" + relation_triple[0] + "\", \"target\": \"" + relation_triple[2] + \
                              "\", \"value\": 5, \"text\": \"" + relation_triple_text_dict[
                                  (relation_triple[0], relation_triple[2])] + "\"},\n")
        else:
            data_export.write("{\"source\": \"" + relation_triple[0] + "\", \"target\": \"" + relation_triple[
                2] + "\", \"value\": 5}\n]\n}")

    # for relation_triple_index, relation_triple in enumerate(relation_triple_list):
    #     if relation_triple_index != len(relation_triple_list) - 1:
    #         data_export.write("{\"source\": \"" + relation_triple[0] + "\", \"target\": \"" + relation_triple[
    #             2] + "\", \"value\": 5},\n")
    #     else:
    #         data_export.write("{\"source\": \"" + relation_triple[0] + "\", \"target\": \"" + relation_triple[
    #             2] + "\", \"value\": 5}\n]\n}")

