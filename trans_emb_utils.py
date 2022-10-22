### use py38
#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import os
from functools import reduce
import pathlib
#import seaborn as sns
from copy import deepcopy
import jionlp as jio
#from zhon import hanzi
#from zhon.hanzi import non_stops
#import zhon
import networkx as nx

import argparse
import json
import re
from collections import defaultdict

import numpy as np
#from fuzzywuzzy import fuzz
from tqdm import tqdm
#import dedupe

from easynmt import EasyNMT

import jieba.posseg as posseg

# In[2]:


import os
import csv
import re
import logging
import optparse

#import dedupe
#from unidecode import unidecode
import shutil

from functools import lru_cache
from timer import timer

from sentence_transformers import SentenceTransformer
from time import time

from urllib.request import urlopen

import json

import requests

import jieba
import pickle as pkl

##### Give json output
#### search with post
port = "8899"
url_format = "http://localhost:{}/{}/{}"
search_task = "search"
#### add with post
add_task = "add"

searcht_task = "search_trans"
#### add with post
addt_task = "add_trans"
#### pass

def load_pickle(path):
    with open(path, "rb") as f:
        return pkl.load(f)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        return pkl.dump(obj, f)


@timer()
def batch_as_list(a, batch_size = int(100000)):
    req = []
    for ele in a:
        if not req:
            req.append([])
        if len(req[-1]) < batch_size:
            req[-1].append(ele)
        else:
            req.append([])
            req[-1].append(ele)
    return req

@timer()
def embedding_produce_wrapper(input_sent_list, model, pool = None,
    batch = True, batch_threshold = 1000,
):
    def sent_emb_to_dict(req ,emb, exists_dict):
        if not req:
            assert emb is None
            return exists_dict
        if hasattr(emb, "numpy"):
            emb = emb.numpy()
        sent_array_map = map(lambda t2:(req[t2[0]], emb[t2[0]]) ,enumerate(req))
        final = list(sent_array_map) + list(exists_dict.items())
        final_dict = dict(final)
        return final_dict

    req = []
    exists_dict = {}

    if not batch:
        for ele in input_sent_list:
            #url = url_format.format(search_task, ele)
            url = url_format.format(port ,search_task, "")[:-1]
            response = requests.post(url, data = {'k_list': json.dumps([k])})

            #search_result = urlopen(url).read()
            search_result = response.content
            search_json = json.loads(search_result)
            '''
            {
            "embedding": [None or np.array obj] (list of, ) same length with k_list
            }
            '''
            assert "embedding" in search_json
            embedding = search_json["embedding"][0]
            assert hasattr(ele, "__len__")
            if not len(embedding):
                req.append(ele)
            else:
                #assert hasattr(ele, "__len__")
                exists_dict[ele] = embedding
    else:
        url = url_format.format(port ,search_task, "")[:-1]
        response = requests.post(url, data = {'k_list': json.dumps(input_sent_list)})
        #search_result = urlopen(url).read()
        search_result = response.content
        search_json = json.loads(search_result)
        embedding_list = search_json["embedding"]
        assert len(embedding_list) == len(input_sent_list)
        for idx ,embedding in enumerate(embedding_list):
            ele = input_sent_list[idx]
            assert hasattr(ele, "__len__")
            if not len(embedding):
                req.append(ele)
            else:
                #assert hasattr(ele, "__len__")
                exists_dict[ele] = embedding

    if pool is None or len(req) <= batch_threshold:
        if req:
            embedding = model.encode(req)
        else:
            embedding = None
    else:
        if req:
            embedding = model.encode_multi_process(req, pool, chunk_size=50)
        else:
            embedding = None

    final_dict = sent_emb_to_dict(req ,embedding, exists_dict)
    #print("final_dict :", final_dict)

    ##### this also should add check in server side.
    add_dict = dict(filter(lambda t2: t2[0] not in exists_dict, final_dict.items()))

    #print("add_dict :", add_dict)

    if not batch:
        for k, v in add_dict.items():
            url = url_format.format(port ,add_task, "")[:-1]
            assert hasttr(v, "size")
            post_content = v.tolist()
            #### check if not exists then add (insert)
            requests.post(url, data = {'k_list': json.dumps([k]), 'arr_list': json.dumps([post_content])})
    else:
        url = url_format.format(port ,add_task, "")[:-1]
        k_list = []
        arr_list = []
        for k, v in add_dict.items():
            assert hasattr(v, "size")
            k_list.append(k)
            post_content = v.tolist()
            arr_list.append(post_content)

        ##### limit every add section DATA_UPLOAD_MAX_MEMORY_SIZE
        for k_, arr_ in zip(
        batch_as_list(k_list, batch_threshold),
        batch_as_list(arr_list, batch_threshold)
        ):
            requests.post(url, data = {'k_list': json.dumps(k_), 'arr_list': json.dumps(arr_)})
        #requests.post(url, data = {'k_list': json.dumps(k_list), 'arr_list': json.dumps(arr_list)})

    l = []
    for ele in input_sent_list:
        l.append(final_dict[ele])
    stack_emb = np.vstack(l)
    assert len(stack_emb.shape) == 2 and stack_emb.shape[0] == len(input_sent_list)
    return stack_emb

@timer()
def repeat_to_one_f(x):
    req = None
    for token in jieba.lcut(x):
        #print("req :", req)

        if len(set(token)) == 1:
            token = token[0]
        if req is None:
            req = token
        else:

            if token in req:
                continue
            else:
                while req.endswith(token[0]):
                    token = token[1:]
                req = req + token
    return req.strip()

@timer()
def repeat_to_one_fb(x):
    return sorted(map(repeat_to_one_f, [x, "".join(jieba.lcut(x)[::-1])]),
                 key = len
                 )[0]

repeat_to_one = repeat_to_one_fb

@timer()
def do_one_trans_produce_wrapper(input_sent_list, model, pool = None,
    batch = True, batch_threshold = 1000,
):
    assert hasattr(model, "translate")

    def sent_emb_to_dict(req ,emb, exists_dict):
        if not req:
            assert emb is None
            return exists_dict
        assert type(emb) == type([])
        sent_array_map = map(lambda t2:(req[t2[0]], emb[t2[0]]) ,enumerate(req))
        final = list(sent_array_map) + list(exists_dict.items())
        final_dict = dict(final)
        return final_dict

    req = []
    exists_dict = {}

    if not batch:
        for ele in input_sent_list:
            #url = url_format.format(search_task, ele)
            url = url_format.format(port ,searcht_task, "")[:-1]
            response = requests.post(url, data = {'k_list': json.dumps([k])})

            #search_result = urlopen(url).read()
            search_result = response.content
            search_json = json.loads(search_result)
            '''
            {
            "embedding": [None or np.array obj] (list of, ) same length with k_list
            }
            '''
            #assert "embedding" in search_json
            assert "trans" in search_json
            #embedding = search_json["embedding"][0]
            embedding = search_json["trans"][0]
            assert type(embedding) == type("")
            assert hasattr(ele, "__len__")
            if not len(embedding):
                req.append(ele)
            else:
                #assert hasattr(ele, "__len__")
                exists_dict[ele] = embedding
    else:
        url = url_format.format(port ,searcht_task, "")[:-1]
        response = requests.post(url, data = {'k_list': json.dumps(input_sent_list)})
        #search_result = urlopen(url).read()
        search_result = response.content
        search_json = json.loads(search_result)
        #embedding_list = search_json["embedding"]
        embedding_list = search_json["trans"]
        assert len(embedding_list) == len(input_sent_list)
        for idx ,embedding in enumerate(embedding_list):
            ele = input_sent_list[idx]
            assert hasattr(ele, "__len__")
            assert type(embedding) == type("")
            if not len(embedding):
                req.append(ele)
            else:
                #assert hasattr(ele, "__len__")
                exists_dict[ele] = embedding


    if pool is None or len(req) <= batch_threshold:
        if req:
            trans_list = model.translate(req,
                   source_lang="en", target_lang = "zh")
            trans_list = list(map(lambda x: repeat_to_one(x) if x else x ,trans_list))
        else:
            trans_list = None
    else:
        # model.translate_multi_process(process_pool,
        # sentences, source_lang='en', target_lang='de', show_progress_bar=True)
        if req:
            trans_list = model.translate_multi_process(pool ,req,
                   source_lang="en", target_lang = "zh")
            trans_list = list(map(lambda x: repeat_to_one(x) if x else x ,trans_list))
        else:
            trans_list = None

    #final_dict = sent_emb_to_dict(req ,embedding, exists_dict)
    final_dict = sent_emb_to_dict(req ,trans_list, exists_dict)
    #print("final_dict :", final_dict)

    ##### this also should add check in server side.
    add_dict = dict(filter(lambda t2: t2[0] not in exists_dict, final_dict.items()))

    #print("add_dict :", add_dict)

    if not batch:
        for k, v in add_dict.items():
            url = url_format.format(port ,addt_task, "")[:-1]
            assert type(v) == type("")

            post_content = v
            #### check if not exists then add (insert)
            requests.post(url, data = {'k_list': json.dumps([k]), 'arr_list': json.dumps([post_content])})
    else:
        url = url_format.format(port ,addt_task, "")[:-1]
        k_list = []
        arr_list = []
        for k, v in add_dict.items():
            #assert hasattr(v, "size")
            assert type(v) == type("")
            k_list.append(k)
            #post_content = v.tolist()
            post_content = v
            arr_list.append(post_content)

        ##### limit every add section DATA_UPLOAD_MAX_MEMORY_SIZE
        for k_, arr_ in zip(
        batch_as_list(k_list, batch_threshold),
        batch_as_list(arr_list, batch_threshold)
        ):
            requests.post(url, data = {'k_list': json.dumps(k_), 'arr_list': json.dumps(arr_)})
        #requests.post(url, data = {'k_list': json.dumps(k_list), 'arr_list': json.dumps(arr_list)})

    l = []
    for ele in input_sent_list:
        l.append(final_dict[ele])
    assert len(l) == len(input_sent_list)
    return l

@timer()
def find_max_len_cut_b_with_entity_maintain_j(a, b, b_entity_list):
    assert type(b_entity_list) == type([])
    ner = jio.ner.LexiconNER(
        {"Un_tokenize": list(set(b_entity_list))}
    )
    b_nered = ner(b)
    def offset_len_split(b ,b_bered):
        if not b_bered:
            return [0, len(b)]
        req = reduce(lambda a, b : a + b ,map(lambda x: x["offset"], b_bered))
        return sorted(set(req + [0, len(b)]))
    offset_indexes = offset_len_split(b, b_nered)
    offset_indexes_nested = []
    for i in range(len(offset_indexes) - 1):
        offset_indexes_nested.append(
            (offset_indexes[i], offset_indexes[i + 1])
        )

    def change_bucket(b_sliced):
        assert type(b_sliced) == type("")
        #print("b_scliced: {} {}".format(b_sliced, len(b_sliced)))
        if len(b_sliced) >= 8:
            bucket_func=lambda x: ["".join(x)]
        elif len(b_sliced) >= 4:
            bucket_func = lambda x: jieba.lcut("".join(x))
        else:
            bucket_func = lambda x: x
        return bucket_func

    #print("offset_indexes_nested :")
    #print(offset_indexes_nested)

    slice_map = map(lambda t2: [b[t2[0]: t2[1]]]
        if b[t2[0]: t2[1]] in b_entity_list else
        jieba.lcut("".join(b[t2[0]: t2[1]]))
        , offset_indexes_nested)
    slice_reduced = reduce(lambda a, b: a + b, slice_map)

    return b_nered, slice_reduced


@timer()
def batch_as_list(a, batch_size = int(100000)):
    req = []
    for ele in a:
        if not req:
            req.append([])
        if len(req[-1]) < batch_size:
            req[-1].append(ele)
        else:
            req.append([])
            req[-1].append(ele)
    return req
