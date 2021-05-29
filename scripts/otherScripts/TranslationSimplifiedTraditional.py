# -*- coding: utf8 -*-
"""
Reference:
* https://github.com/gumblex/zhconv

"""
from tqdm import tqdm
from zhconv import convert
import codecs
import os

if __name__ == '__main__':
    # DATA_IMPORT_DIR = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetSimplified/SentenceIncident/"
    # DATA_EXPORT_DIR = "../../../KnowledgeGraph_materials/data_kg/baiduDatasetTraditional/SentenceIncident/"
    DATA_IMPORT_DIR = "C:/Users/User/Desktop/tempImport/"
    DATA_EXPORT_DIR = "C:/Users/User/Desktop/tempExport/"

    ''' process starts '''
    for fileIndex, fileElement in enumerate(tqdm(os.listdir(DATA_IMPORT_DIR))):

        if fileElement[-4:] in [".zip", "docx", ".pdf"] or fileElement == "__MACOSX":
            continue

        try:
            for subfileindex, subfileElement in enumerate(os.listdir(DATA_IMPORT_DIR + fileElement)):

                if subfileElement[-4:] in [".zip", "docx", ".pdf"] or subfileElement == "__MACOSX":
                    continue

                data_import = codecs.open(DATA_IMPORT_DIR + fileElement + "/" + subfileElement, "r", encoding="utf8", errors="ignore")
                data_export = codecs.open(DATA_EXPORT_DIR + fileElement + "/" + subfileElement, "w", encoding="utf8")

                lines = data_import.readlines()
                for lineIndex, lineElement in enumerate(lines):
                    lines[lineIndex] = convert(lineElement, 'zh-hant')

                data_export.writelines(lines)
                data_import.close()
                data_export.close()

        except NotADirectoryError:
            data_import = codecs.open(DATA_IMPORT_DIR + fileElement, "r", encoding="utf8", errors="ignore")
            data_export = codecs.open(DATA_EXPORT_DIR + fileElement, "w", encoding="utf8")

            lines = data_import.readlines()
            for lineIndex, lineElement in enumerate(lines):
                lines[lineIndex] = convert(lineElement, 'zh-hant')

            data_export.writelines(lines)
            data_import.close()
            data_export.close()
