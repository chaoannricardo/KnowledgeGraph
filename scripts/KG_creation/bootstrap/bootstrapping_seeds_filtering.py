# -*- encoding: utf8 -*-
import codecs
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    ''' Configurations '''
    MATERIALS_DIR = "C:/Users/User/Desktop/Ricardo/KnowledgeGraph_materials/"
    SEED_IMPORT_PATH = MATERIALS_DIR + "data_kg/baiduDatasetTranditional_Cleansed/seed_relations_basic.csv"
    FILTER_SEED_EXPORT_PATH =  MATERIALS_DIR + "data_kg/baiduDatasetTranditional_Cleansed/seed_relations_basic_filtered.csv"
    FILTER_COUNT_CRITEREA = 14
    IF_SHOW_ESTIMATE_CURVE = True
    STARTING_FILTER = 5

    ''' Process Starts '''
    seed_pattern_list = []

    seed_basic = codecs.open(SEED_IMPORT_PATH, mode="r", encoding="utf8", errors="ignore")
    seed_filter_export = codecs.open(FILTER_SEED_EXPORT_PATH, mode="w", encoding="utf8", errors="ignore")
    starting_filter = STARTING_FILTER if IF_SHOW_ESTIMATE_CURVE else FILTER_COUNT_CRITEREA
    filter_num_list = []
    filter_seed_count_list = []

    for seedIndex, seedElement in enumerate(seed_basic.readlines()):
        original_pattern = seedElement.split("&")[0]
        relation = seedElement.split("&")[1]
        original_pattern = original_pattern.replace(relation.replace("\n", ""), "Predicate")
        seed_pattern_list.append(original_pattern)

    # initiate object to prevent warning
    data_filter = None
    for iteration in range(starting_filter, FILTER_COUNT_CRITEREA):
        # calculate and filter out import seeds
        data_calculate = pd.DataFrame({
            "SeedType": seed_pattern_list
        })

        data_filter = data_calculate["SeedType"].value_counts().reset_index()
        data_filter = data_filter[data_filter["SeedType"] >= iteration].iloc[:, 0].tolist()
        filter_num_list.append(iteration)
        filter_seed_count_list.append(len(data_filter))

    for lineIndex, lineElement in enumerate(data_filter):
        seed_filter_export.write(lineElement + "\n")

    print(filter_num_list)
    print(filter_seed_count_list)

    if IF_SHOW_ESTIMATE_CURVE:
        plt.bar(filter_num_list, filter_seed_count_list)
        plt.xlabel("Filter Threshold")
        plt.xticks(filter_num_list)
        plt.ylabel("Unique Seed Types")
        plt.title("Bootstrapping Seed Filtering")

        # https://www.python-graph-gallery.com/10-barplot-with-number-of-observation
        for i in range(len(filter_seed_count_list)):
            plt.text(x=filter_num_list[i]-0.2, y=filter_seed_count_list[i] + 0.5,
                     s=filter_seed_count_list[i], size=10, c="navy",
                     fontweight="bold")


        plt.show()
















