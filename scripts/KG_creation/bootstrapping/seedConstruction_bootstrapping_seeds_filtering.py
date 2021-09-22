# -*- encoding: utf8 -*-
from tqdm import tqdm
import codecs
import matplotlib.pyplot as plt
import pandas as pd

''' Configurations '''
MATERIALS_DIR = "C:/Users/User/Desktop/Ricardo/KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/"
SEED_IMPORT_PATH = MATERIALS_DIR + "seed_relations.csv"
SEED_BASIC_IMPORT_PATH = MATERIALS_DIR + "seed_relations_basic.csv"
FILTER_SEED_BASIC_EXPORT_PATH = MATERIALS_DIR + "seed_relations_basic_filtered.csv"
FILTER_SEED_EXPORT_PATH = MATERIALS_DIR + "seed_relations_filtered.csv"
FILTER_COUNT_CRITERIA = 10
IF_SHOW_ESTIMATE_CURVE = True
STARTING_FILTER = 5

if __name__ == '__main__':
    # list to store seed pattern
    seed_pattern_list = []

    # read in seeds
    seed = codecs.open(SEED_IMPORT_PATH, mode="r", encoding="utf8", errors="ignore")
    seed_basic = codecs.open(SEED_BASIC_IMPORT_PATH, mode="r", encoding="utf8", errors="ignore")

    # create file to write
    seed_filter_export = codecs.open(FILTER_SEED_EXPORT_PATH, mode="w", encoding="utf8", errors="ignore")
    seed_filter_basic_export = codecs.open(FILTER_SEED_BASIC_EXPORT_PATH, mode="w", encoding="utf8", errors="ignore")

    # plot usage: showing seeds count from what number of fileter
    starting_filter = STARTING_FILTER if IF_SHOW_ESTIMATE_CURVE else FILTER_COUNT_CRITERIA
    filter_num_list = []
    filter_seed_count_list = []

    # mask out each pattern's predicate and store into list
    for seedIndex, seedElement in enumerate(seed_basic.readlines()):
        original_pattern = seedElement.split("&")[0]
        relation = seedElement.split("&")[1]
        original_pattern = original_pattern.replace(relation.replace("\n", ""), "Predicate")
        seed_pattern_list.append(original_pattern)

    # initiate object to prevent warning
    # filter seed by iteration, in order to create plot
    data_filter = None
    for iteration in range(starting_filter, FILTER_COUNT_CRITERIA):
        # calculate and filter out import seeds
        data_calculate = pd.DataFrame({
            "SeedType": seed_pattern_list
        })

        data_filter = data_calculate["SeedType"].value_counts().reset_index()
        data_filter = data_filter[data_filter["SeedType"] >= iteration].iloc[:, 0].tolist()
        filter_num_list.append(iteration)
        filter_seed_count_list.append(len(data_filter))

    # write filtered seed into files
    for lineIndex, lineElement in enumerate(data_filter):
        seed_filter_basic_export.write(lineElement + "\n")

    print("Filter Threshold", filter_num_list)
    print("Seed Type Count", filter_seed_count_list)

    # filter original seed and export
    print("Now filtering original seeds...")
    for lineIndex, line in enumerate(tqdm(seed.readlines())):
        try:
            line = line.replace("\n", "")
            edge = line.split("&")[-1]
            dependency_node_list = "&".join(line.split("&")[:-1]).split("@")
            dependency_node_list[0] = "Entity"
            dependency_node_list[-1] = "Entity"
            dependency_node_list[dependency_node_list.index(edge)] = "Predicate"
            basic_pattern = "@".join(dependency_node_list)

            if basic_pattern in data_filter:
                seed_filter_export.write(line + "\n")
        except ValueError:
            continue

    # plotting filtering curve
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
















