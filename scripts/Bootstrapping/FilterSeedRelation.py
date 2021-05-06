# -*- encoding: utf8 -*-
import codecs
import pandas as pd


if __name__ == '__main__':
    SEED_IMPORT_PATH = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_Cleansed/SEED_RELATION_BASIC.csv"


    seed_pattern_list = []

    seed_basic = codecs.open(SEED_IMPORT_PATH, mode="r", encoding="utf8", errors="ignore")

    for seedIndex, seedElement in enumerate(seed_basic.readlines()):
        original_pattern = seedElement.split("&")[0]
        relation = seedElement.split("&")[1]
        original_pattern = original_pattern.replace(relation.replace("\n", ""), "Predicate")
        seed_pattern_list.append(original_pattern)

    # calculate and filter out import seeds
    data_calculate = pd.DataFrame({
        "SeedType": seed_pattern_list
    })

    print(data_calculate["SeedType"].value_counts())

    # print(len(data_calculate))
    #
    # print(len(data_calculate.drop_duplicates()))
    # print(data_calculate.groupby("SeedType").count())



