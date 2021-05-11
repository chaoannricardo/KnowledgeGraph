# -*- coding: utf8 -*-
from google_trans_new import google_translator
from tqdm import tqdm
import codecs
import google_trans_new
import time

if __name__ == '__main__':
    ''' Configuration '''
    NUMBER_LIST = [str(i) for i in range(10000)]
    RAW_DATA_PATH = "../dicts/SemiconductorDict_Raw.txt"
    ENGLISH_DICT_PATH = "../dicts/SemiconductorDict_ENGLISH.txt"
    MANDARIN_DICT_PATH = "../dicts/SemiconductorDict_MANDARIN.txt"
    REPLACE_CHAR = ["\r\n", "\n"]
    ''' Configuration Ended '''
    dict_export_list = []
    dict_translated_list = []
    file_original = codecs.open(RAW_DATA_PATH, mode="r", encoding="utf8", errors="ignore")
    file_export_english = codecs.open(ENGLISH_DICT_PATH, mode="w", encoding="utf8")
    file_export_mandarin = codecs.open(MANDARIN_DICT_PATH, mode="w", encoding="utf8")
    translator = google_translator()

    for lineIndex, lineElement in enumerate(tqdm(file_original.readlines())):
        line_split_list = lineElement.split(",")
        for splitIndex, splitElement in enumerate(line_split_list):
            if splitElement[0] == " ":
                splitElement = splitElement[1:]
            elif splitElement[-1] == " ":
                splitElement = splitElement[:-1]

            # split element with page dash break
            sub_line_split_list = splitElement.split("–")
            needEliminate = False
            for subsplitIndex, subsplitElement in enumerate(sub_line_split_list):
                for replaceIndex, replaceElement in enumerate(REPLACE_CHAR):
                    subsplitElement = subsplitElement.replace(replaceElement, "")

                if subsplitElement in NUMBER_LIST:
                    needEliminate = True
                    break

            if not needEliminate:
                for replaceIndex, replaceElement in enumerate(REPLACE_CHAR):
                    splitElement = splitElement.replace(replaceElement, "")

                dict_export_list.append(splitElement)
                file_export_english.write(splitElement + "\n")

    print("Start Translating...")

    for element in tqdm(dict_export_list):
        while True:
            try:
                translation = translator.translate(element, lang_src="en", lang_tgt="zh-tw")
                dict_translated_list.append(translation)
                file_export_mandarin.write(translation + "\n")
                break
            except google_trans_new.google_trans_new.google_new_transError:
                time.sleep(60)
                pass








