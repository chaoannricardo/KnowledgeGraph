# -*- coding: utf8 -*-
from gensim.models import word2vec
from tqdm import tqdm
import numpy as np
import pandas as pd

'''
# other functions
print(model.wv.similarity("晶圓", "價值"))
'''

if __name__ == '__main__':
    # configurations
    ENTITY_LOAD_PATH = "../../results/210408_result/210408_dataset_entity_result_MONPA.csv"
    MODEL_PATH = "../../models/210408_word2vec.model"
    SAVE_FILE_PATH = "../../results/210408_result/210408_dataset_word2vec_kg.csv"

    # process starts
    model = word2vec.Word2Vec.load(MODEL_PATH)
    data_entities = pd.read_csv(ENTITY_LOAD_PATH, sep=",")
    entity_list = data_entities.iloc[:, 0].tolist()
    # print(model.wv.most_similar([entity_list[0], entity_list[1]]))

    print("-- Process Starts --")
    saving_file = open(SAVE_FILE_PATH, "w", encoding="utf8")
    header_text = "entity_0,entity_1,relation_0,similarity_0,relation_1,similarity_1,relation_2,similarity_2," + \
        "relation_3,similarity_3,relation_4,similarity_4,relation_5,similarity_5,relation_6,similarity_6" +\
        "relation_7,similarity_7,relation_8,similarity_8,relation_9,similarity_9\n"
    saving_file.write(header_text)
    for entityIndex, entityElement in enumerate(tqdm(entity_list)):
        for targetIndex in range((entityIndex+1), (len(entity_list))):
            try:
                model.wv.most_similar([entityElement, entity_list[targetIndex]])
                saving_file.write(str(entityElement) + "," + str(entity_list[targetIndex]) + ",")
                for relationIndex, relationElement in enumerate(
                        model.wv.most_similar([entityElement, entity_list[targetIndex]])):
                    if relationIndex + 1 != (len(model.wv.most_similar([entityElement, entity_list[targetIndex]]))):
                        saving_file.write(str(relationElement[0]) + ",")
                        saving_file.write(str(np.around(relationElement[1], decimals=4)) + ",")
                    else:
                        saving_file.write(str(relationElement[0]) + ",")
                        saving_file.write(str(np.around(relationElement[1], decimals=4)) + "\n")
            except KeyError:
                pass

    saving_file.close()

