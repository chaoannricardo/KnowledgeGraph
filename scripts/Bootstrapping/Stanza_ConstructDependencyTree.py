# -*- coding: utf8 -*-
"""
Reference: https://stackoverflow.com/questions/36610179/how-to-get-the-dependency-tree-with-spacy

Other Reference related to dependency graph construction:
* displaCy Dependency Visualizer Online: https://reurl.cc/E20jgK
* https://spacy.io/api/dependencyparser
* https://spacy.io/usage/visualizers


spacy.displacy.serve(doc, style='dep')
"""
from nltk import Tree
import stanza



def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


if __name__ == '__main__':
    ''' Configurations '''

    config = {
        'processors': 'tokenize,pos,lemma,depparse',  # Comma-separated list of processors to use
        'lang': 'zh',  # Language code for the language to build the Pipeline in
        'tokenize_model_path': '../../../KnowledgeGraph_materials/stanza_resources/zh-hant/tokenize/gsd.pt',
        # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
        'pos_model_path': '../../../KnowledgeGraph_materials/stanza_resources/zh-hant/pos/gsd.pt',
        'pos_pretrain_path': '../../../KnowledgeGraph_materials/stanza_resources/zh-hant/pretrain/gsd.pt',
        'lemma_model_path': '../../../KnowledgeGraph_materials/stanza_resources/zh-hant/lemma/gsd.pt',
        'depparse_model_path': '../../../KnowledgeGraph_materials/stanza_resources/zh-hant/depparse/gsd.pt',
        'depparse_pretrain_path': '../../../KnowledgeGraph_materials/stanza_resources/zh-hant/pretrain/gsd.pt',
    }

    ''' Configurations Ended '''

    allWords = []
    allFlags = []
    allDep = []

    text = "晶圓代工廠商的先進製程競賽如火如荼來到7nm，但也有晶圓代工廠商就此打住，聯電將止於12nm製程研發，GlobalFoundries宣告無限期停止7nm及以下先進製程發展。"

    nlp = stanza.Pipeline(**config) # Initialize the pipeline using a configuration dict
    doc = nlp(text)

    # for token in doc:
    #     allWords.append(token.text)
    #     allFlags.append(token.pos_)
    #     allDep.append(token.dep_)

    # print(allWords)
    # print(allFlags)
    # print(allDep)
    #
    # print([to_nltk_tree(sent.root) for sent in doc.sents])

    # print([sent.words[word.head - 1].text if word.head > 0 else "root" for sent in doc.sentences for word in sent.words])

    print([to_nltk_tree(sent.words[word.head - 1].text) if word.head > 0 else "root" for sent in doc.sentences for word in sent.words])

