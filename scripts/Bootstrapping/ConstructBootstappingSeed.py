# -*- coding: utf8 -*-
from nltk import Tree
import networkx as nx
import re
import spacy


def find_element_nested_list(element, list_source, index_list, index="start"):
    # find element recursively
    for the_index, subList in enumerate(list_source):
        if type(subList) is not list:
            if list_source[the_index] == element:
                if index != "start": index_list.append(str(index))
                index_list.append(str(the_index))
                index_list.append("@")
        else:
            find_element_nested_list(element, subList, index_list, the_index)

    index_result = "-".join(index_list)
    index_result = re.split("|".join(["-@-", "-@"]), index_result)[:-1]

    return index_result


if __name__ == '__main__':
    SEED_OUTPUT_PATH = ""
    