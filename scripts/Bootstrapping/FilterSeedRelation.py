# -*- encoding: utf8 -*-
import codecs
import pandas as pd


if __name__ == '__main__':
    ''' Configurations '''
    SEED_IMPORT_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_RELATION_BASIC.csv"
    FILTER_SEED_EXPORT_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_RELATION_BASIC_FILTER.csv"
    FILTER_COUNT_CRITEREA = 6
    ''' Process Starts '''
    seed_pattern_list = []

    seed_basic = codecs.open(SEED_IMPORT_PATH, mode="r", encoding="utf8", errors="ignore")
    seed_filter_export = codecs.open(FILTER_SEED_EXPORT_PATH, mode="w", encoding="utf8", errors="ignore")

    for seedIndex, seedElement in enumerate(seed_basic.readlines()):
        original_pattern = seedElement.split("&")[0]
        relation = seedElement.split("&")[1]
        original_pattern = original_pattern.replace(relation.replace("\n", ""), "Predicate")
        seed_pattern_list.append(original_pattern)

    # calculate and filter out import seeds
    data_calculate = pd.DataFrame({
        "SeedType": seed_pattern_list
    })

    data_filter = data_calculate["SeedType"].value_counts().reset_index()
    data_filter = data_filter[data_filter["SeedType"] >= FILTER_COUNT_CRITEREA].iloc[:, 0].tolist()

    for lineIndex, lineElement in enumerate(data_filter):
        seed_filter_export.write(lineElement + "\n")

    # print(len(data_filter[data_filter["SeedType"] >= 5]))








