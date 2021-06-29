# -*- coding: utf8 -*-
import codecs
import os
import pandas as pd
import tqdm


if __name__ == '__main__':
    ''' Configurations '''
    # Word Chronology
    # small dataset
    # STRICT_RELATION_TRIPLES_PATH = "C:\\Users\\User\\Desktop\\Ricardo\\KnowledgeGraph_materials\\results_kg\\WorldChronology\\STRICT_SEED_RELATION_WHOLE.csv"
    # DATA_TEXT_DIR_PATH = "C:\\Users\\User\\Desktop\\Ricardo\\KnowledgeGraph_materials\\data_kg\\WorldChronologyMandarin"
    # OUTPUT_PATH = "C:\\Users\\User\\Desktop\\Ricardo\\KnowledgeGraph_materials\\results_kg\\WorldChronology\\STRICT_SEED_RELATION_RANKING.csv"

    # Semiconductor
    # small dataset
    # STRICT_RELATION_TRIPLES_PATH = "C:\\Users\\User\\Desktop\\Ricardo\\KnowledgeGraph_materials\\results_kg\\Semiconductor\\STRICT_SEED_RELATION_WHOLE.csv"
    # DATA_TEXT_DIR_PATH = "C:\\Users\\User\\Desktop\\Ricardo\\KnowledgeGraph_materials\\data_kg\\data_normal_wafer_text"
    # OUTPUT_PATH = "C:\\Users\\User\\Desktop\\Ricardo\\KnowledgeGraph_materials\\results_kg\\Semiconductor\\STRICT_SEED_RELATION_RANKING.csv"
    # large dataset
    STRICT_RELATION_TRIPLES_PATH = "C:\\Users\\User\\Desktop\\Ricardo\\KnowledgeGraph_materials\\results_kg\\SemiconductorAll\\STRICT_SEED_RELATION_WHOLE.csv"
    DATA_TEXT_DIR_PATH = "C:\\Users\\User\\Desktop\\Ricardo\\KnowledgeGraph_materials\\data_kg\\data_normal_wafer_textAll"
    OUTPUT_PATH = "C:\\Users\\User\\Desktop\\Ricardo\\KnowledgeGraph_materials\\results_kg\\SemiconductorAll\\STRICT_SEED_RELATION_RANKING.csv"

    ''' Process Starts '''
    # load data
    relation_triples_file = codecs.open(STRICT_RELATION_TRIPLES_PATH, encoding="utf8", errors="ignore")

    # changing working dir
    mother_dir = os.getcwd()
    os.chdir(DATA_TEXT_DIR_PATH)

    # initiate dict to store data, relation list
    result = {}
    relation_triple_list = []
    for relationTriples in relation_triples_file.readlines():
        relationTriples = relationTriples.replace("\r\n", "").replace("\n", "")
        result[relationTriples] = 0
        relation_triple_list.append(relationTriples.split("@"))

    # enumerate over all text files
    for fileindex, fileElement in enumerate(os.listdir()):
        text_file = codecs.open(fileElement, encoding="utf8", errors="ignore")
        for lineIndex, line in enumerate(text_file.readlines()):
            # enumerate all relations
            for relationTriples in relation_triple_list:
                if relationTriples[0] in line and relationTriples[1] in line and relationTriples[2] in line:
                    result["@".join(relationTriples)] += 1

    # construct dataframe with dict & sort by count
    data_result = pd.DataFrame(list(result.items()),
                               columns=["relationTriple", "count"])
    data_result.sort_values(by="count", ascending=False, ignore_index=True, inplace=True)

    # export result
    data_result.to_csv(OUTPUT_PATH, index=False, header=True, encoding="utf8", sep=",")
    print(data_result)

