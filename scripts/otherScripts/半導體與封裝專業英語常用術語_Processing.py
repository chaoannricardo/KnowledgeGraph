# -*- coding: utf8 -*-
from tqdm import tqdm
import codecs
import os


if __name__ == '__main__':
    ''' Configuration '''
    DATA_PATH = "../../../KnowledgeGraph_materials/data_kg/WaferManufacturingDictionarySource/半導體與封裝專業英語常用術語.txt"

    ''' Process Starts '''
    data_import = codecs.open(DATA_PATH, mode="r", encoding="utf8", errors="ignore")
    english_vocab = []
    chinese_vocab = []

    for lineIndex, lineElement in enumerate(tqdm(data_import.readlines())):

        # print(lineElement)

        # skip if not vocabulary lines
        if "：" not in lineElement:
            continue

        main_line = lineElement.split("：")[0]

        # dealing with english part
        english_part = main_line.split("/")[0]
        if "（" in english_part and "）" in english_part:
            english_vocab.append(english_part.split("（")[0])
            # dealing with remain part
            english_vocab.append(english_part.split("（")[1].replace("）", ""))
        else:
            english_vocab.append(english_part)

        # dealing with chinese part
        chinese_part = main_line.split("/")[1]
        if "（" in chinese_part and "）" in chinese_part:
            chinese_vocab.append(chinese_part.split("（")[0])
            # dealing with remain part
            chinese_vocab.append(chinese_part.split("（")[1].replace("）", ""))
        else:
            chinese_vocab.append(chinese_part)

    print(english_vocab)
    print(chinese_vocab)

    for vocabIndex, vocabElement in enumerate(english_vocab + chinese_vocab):
        print(vocabElement)

        

