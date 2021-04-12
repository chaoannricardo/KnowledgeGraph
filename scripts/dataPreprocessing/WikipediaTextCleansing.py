# -*- coding: utf8 -*-
"""

Python 轻量化简繁转换 zhconv
* https://github.com/gumblex/zhconv
* https://zhuanlan.zhihu.com/p/55973055
"""
from tqdm import tqdm
from zhconv import convert
import codecs
import os
import re

if __name__ == '__main__':
    # configurations
    DATA_IMPORT_PATH = "../../data/data_wikipedia_xml/"
    DATA_EXPORT_PATH = "../../data/data_wikipedia_xml_clean/"
    NEGLECT_SYMBOLS = ["[", "]", "*", "'", "=", "●", "}", "{", "/", "|"]
    NEGLECT_HTTP_TYPE = ["http", "https", "styleborder:", "text-align:", "cellspacing", "Wikipedia:"]
    NEGLECT_SUBSTRING = ["&lt;", "ref&gt;", "ref", "OED&quot;", "&gt;", "name&quot;", "{{Cite web",\
                         "lamberg-karlovsky-p5&quot;", ":3&quot;" "zbjn1&quot;", "&quot;", "Category:", "solid",\
                         "#ddd", "margin: auto;", "name", "KneeboneCite", "book", "LaTorreCite", "cite journal", "zh",\
                         "amencps-21", "groupa", "brencps-21", "cnencps-21", "cntwencps-21"] + NEGLECT_HTTP_TYPE
    NEGLECT_LINE_SYMBOL = ["|", "{", "\n", " ", "_", "#", ":"]
    # # Common, usually entity100)]


    # process starts
    for fileIndex, fileElement in enumerate(os.listdir(DATA_IMPORT_PATH)):
        fileImport = codecs.open(DATA_IMPORT_PATH + fileElement, "r", encoding="utf8", errors="ignore")
        fileExport = codecs.open(DATA_EXPORT_PATH + fileElement, "w", encoding="utf8", errors="ignore")
        for line in tqdm(fileImport):
            # skip line if fulfill certain sentence structure
            if line[0:4] == "&lt;" or line[0:5] == "&amp;" or line[0:2] == " |" or line[0:3] == " | " or line[0] == " " or\
                    line[0:7] == "[[File:" or len(line) < 20 or line[0] in NEGLECT_LINE_SYMBOL:
                continue
            # check if characters fulfill certain structure
            for characterIndex, character in enumerate(line):
                # eliminate html labels
                if character == "<":
                    for findingIndex in range(characterIndex, len(line)):
                        if line[findingIndex] == ">":
                            line_list = list(line)
                            line_list[characterIndex:(findingIndex+1)] = ["●" for i in range((findingIndex+1) - characterIndex)]
                            line = "".join(line_list)
                            break
                elif line[characterIndex:characterIndex+2] == "[[":
                    for findingIndex in range(characterIndex, len(line)):
                        if line[findingIndex:findingIndex+2] == "]]":
                            temp_list = line[characterIndex+2:findingIndex]
                            try:
                                splitter_index = temp_list.index("|")
                                line_list = list(line)
                                line_list[characterIndex:characterIndex + 2] = ["●", "●"]
                                line_list[(characterIndex + 2 + splitter_index):findingIndex] = ["●" for i in range(findingIndex - (characterIndex + 2 + splitter_index))]
                                line = "".join(line_list)
                            except ValueError:
                                break
                elif line[characterIndex:characterIndex+2] == "{{":
                    for findingIndex in range(characterIndex, len(line)):
                        if line[findingIndex:findingIndex + 2] == "}}":
                            line_list = list(line)
                            line_list[characterIndex:findingIndex + 2] = ["●" for i in range(findingIndex + 2 - characterIndex)]
                            line = "".join(line_list)
                            break
                elif line[characterIndex] == "[":
                    for findingIndex in range(characterIndex, len(line)):
                        if line[findingIndex] == "]":
                            temp_list = line[characterIndex + 1:findingIndex]
                            try:
                                splitter_index = temp_list.index("|")
                                line_list = list(line)
                                line_list[(characterIndex + 1 + splitter_index):findingIndex] = ["●" for i in range(findingIndex - (characterIndex + 1 + splitter_index))]
                                line_list[characterIndex] = "●"
                                line = "".join(line_list)
                            except ValueError:
                                break

            for neglectWordIndex, neglectWord in enumerate(NEGLECT_SUBSTRING):
                if neglectWord in line:
                    line_list = list(line)
                    for characterIndex, character in enumerate(line_list):
                        if character == neglectWord[0] and ("".join(line_list[characterIndex:(characterIndex + len(neglectWord))])) == neglectWord:
                            if neglectWord not in NEGLECT_HTTP_TYPE:
                                line_list[characterIndex:(characterIndex + len(neglectWord))] = ["●" for i in range(len(neglectWord))]
                            else:
                                # special part to remove website link
                                for findingIndex in range(characterIndex, len(line)):
                                    if line[findingIndex] in [" ", "\n"]:
                                        line_list[characterIndex:findingIndex] = ["●" for i in range(findingIndex - characterIndex)]
                                        break

                    line = "".join(line_list)

            # Final elimination for certain sentence structure
            line_list = list(line)
            for neglectIndex, neglectElement in enumerate(NEGLECT_SYMBOLS):
                line_list = list(filter(neglectElement.__ne__, line_list))

            if set(line_list) == " " or line[0:5] == "File:" or len(line) < 20 or line[0] in NEGLECT_LINE_SYMBOL:
                continue

            line = "".join(line_list)
            line = convert(line, 'zh-hant')

            # write line in export document
            fileExport.write(line)





