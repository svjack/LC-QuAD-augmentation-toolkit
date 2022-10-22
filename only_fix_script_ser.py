from timer import timer, get_timer
import logging

logging.basicConfig(level=logging.DEBUG)

# or you can change default timer's logging level
timer.set_level(logging.DEBUG)

import difflib
import os
from hashlib import sha512

import jieba.posseg as posseg
import seaborn as sns


from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from easynmt import EasyNMT

from sklearn.linear_model import LinearRegression

#### load nodel without nltk download
os.environ["DP_SKIP_NLTK_DOWNLOAD"] = "True"

import numpy as np
import pandas as pd
from deeppavlov import build_model, configs
from deeppavlov.core.commands.train import read_data_by_config
from deeppavlov.core.common.chainer import *

import inspect
from kbqa_entity_linking import *

pd.set_option('display.max_colwidth', -1)

import difflib
import os
import re
from functools import partial, reduce, lru_cache
from hashlib import sha512
from itertools import combinations, permutations, product

import jieba
import jionlp as jio
import spacy
'''
from deeppavlov.core.commands.infer import *
from deeppavlov.dataset_readers.sq_reader import OntonotesReader
from deeppavlov.models.kbqa.entity_detection_parser import *
'''
from rapidfuzz import fuzz
from scipy.special import softmax
from sentence_transformers.util import pytorch_cos_sim

from joblib import Parallel ,delayed

from random import sample

import shutil
import json

from time import time

import sqlite_utils

import cn2an

import opencc

#import stanza
#import spacy_stanza

#import stanza
#import spacy_stanza
import numpy as np
import pandas as pd
import os
import seaborn as sns
#import spacy_conll
#from udapi.block.read.conllu import Conllu
from io import StringIO
#from udapi.core.node import Node
from easynmt import EasyNMT
#from googletrans import Translator

#from conllu import parse

from pathlib import Path

from urllib.request import urlopen

import json

import requests

from copy import deepcopy

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

'''
@timer()
def reverse_list_to_dict(zh_entity_dict, zh_fix_entity_list):
    ori_zh_entity_list = list(set(reduce(lambda a, b: a + b, zh_entity_dict.values()))) if zh_entity_dict else []
    if not ori_zh_entity_list or not zh_fix_entity_list:
        return {}
    ori_fix_dict = dict(map(lambda ori_zh:(ori_zh,sorted(zh_fix_entity_list, key = lambda fix_zh:
    fuzz.ratio(fix_zh, ori_zh),
                             reverse=True
                             )[0]), ori_zh_entity_list))
    return ori_fix_dict
'''

@timer()
def reverse_list_to_dict(zh_entity_dict, zh_fix_entity_list):
    ori_zh_entity_list = list(set(reduce(lambda a, b: a + b, zh_entity_dict.values()))) if zh_entity_dict else []
    if not ori_zh_entity_list or not zh_fix_entity_list:
        return {}
    ori_fix_dict = dict(map(lambda ori_zh:(ori_zh,sorted(zh_fix_entity_list, key = lambda fix_zh:
    fuzz.ratio(fix_zh, ori_zh),
                             reverse=True
                             )[0]), ori_zh_entity_list))
    ori_fix_dict_valid = all(map(lambda t2:
                             t2[0] in t2[1] or t2[1] in t2[0]
                             , ori_fix_dict.items()))
    if ori_fix_dict_valid:
        return ori_fix_dict
    else:
        ori_fix_dict = dict(map(lambda t2: (t2[0], t2[0]), ori_fix_dict.items()))
    ori_fix_dict_valid = all(map(lambda t2:
                             t2[0] in t2[1] or t2[1] in t2[0]
                             , ori_fix_dict.items()))
    assert ori_fix_dict_valid
    return ori_fix_dict

@timer()
def nest_list_to_ref(nest_list):
    return list(map(lambda x: " ".join(filter(lambda y: y.strip(), x)), nest_list))

p_before_join = list(",." + "，。" + ":")
p_len = len(p_before_join)
cp_map = map(lambda i: list(combinations(p_before_join, i)) ,range(1 ,p_len))
cp_list = list(cp_map)
cp_list = reduce(lambda a, b: a + b, cp_list) if cp_list else []

@timer()
def retrieve_sent_split(sent,
                       stops_split_pattern = "|".join(map(lambda x: r"\{}".format(x),
                                                                 ",." + "，。" + ":"))
                       ):
    if not sent.strip():
        return []

    split_list = re.split(stops_split_pattern, sent)
    return split_list

@timer()
def retrieve_all_contain_entity_chips(zh_parse, cp_list, length_threshold = 25):
    sent, entity_dict = parse_en_parse(zh_parse)
    sent_sp_list = []
    for cp in cp_list:
        stops_split_pattern = "|".join(map(lambda x: r"\{}".format(x),
                                                                 "".join(cp)))

        sent_sp_list.append(retrieve_sent_split(sent, stops_split_pattern))
    sent_sp_list = list(set(map(tuple ,sent_sp_list)))
    sent_sp_list = list(map(list, sent_sp_list))
    sent_sp_join_nest_list = []
    for sent_sp in sent_sp_list:
        assert type(sent_sp) == type([])

        sent_sp_join_nest_list.extend(
            produce_join_action(sent_sp)
        )

    sent_sp_join_list = reduce(lambda a, b: a + b, sent_sp_join_nest_list) if sent_sp_join_nest_list else []
    sent_sp_join_list = list(set(sent_sp_join_list))

    entity_list = reduce(lambda a, b: a + b, entity_dict.values()) if entity_dict else []

    info_df = pd.DataFrame(
    pd.Series(sent_sp_join_list).drop_duplicates().map(
        lambda x: (x, find_max_len_cut_b_with_entity_maintain_j("", x, entity_list)[0])
    ).values.tolist()
    , columns = ["sent", "maintain_info"])

    info_df["info_score"] = info_df["maintain_info"].map(
    lambda x: len(x) * int(set(map(lambda d: d["text"], x)) == set(entity_list)) if x else 0
)
    info_df["length"] = info_df.apply(lambda s: len(s["sent"]) - sum(map(lambda d: len(d["text"]) ,s["maintain_info"]))
                                      if s["maintain_info"] else len(s["sent"]), axis = 1)
    #return info_df
    will_return = None
    info_df = info_df.sort_values(by = ["info_score", "length"], ascending = False)
    #return info_df
    for idx, r in info_df.iterrows():
        if r["length"] <= length_threshold and r["info_score"] == info_df["info_score"].max():
            will_return = r["sent"]
    if will_return is None:
        l_c = find_max_len_cut_b_with_entity_maintain_j("", sent, entity_list)[1]
        l_req = []
        for ele in l_c:
            if ele not in entity_list:
                l_req.append(ele)
            else:
                break
        r_req = []
        for ele in l_c[::-1]:
            if ele not in entity_list:
                r_req.append(ele)
            else:
                break
        l = "".join(l_req)
        r = "".join(r_req[::-1])
        assert sent.startswith(l)
        assert sent.endswith(r)
        candidates = (sent[len(l):], sent[:-1 * len(r)], sent[len(l):-1 * len(r)])
        cl = sorted(candidates, key = len)
        #print(sent ,cl, entity_list)
        while cl:
            ### ele = cl.pop()
            ele = cl.pop(0)
            if all(map(lambda x: x in ele, entity_list)):
                c = ele
                break
        assert c in sent
        will_return = c
    for k in entity_list:
        assert k in will_return
    return will_return

@timer()
def produce_join_action(token_list):
    #token_list = list(filter(lambda y: y ,map(lambda x: x.strip().replace(" ", ""), token_list)))
    token_list = list(filter(lambda y: y ,map(lambda x: x.strip(), token_list)))
    if len(token_list) <= 1:
        return [token_list]
    action_nest_zip = product(*map(lambda _: ("+", "/") ,range(len(token_list) - 1)))
    req = []
    for sep_l in action_nest_zip:
        #print(sep_l)
        text = ""
        for i, sep in enumerate(sep_l):
            text += token_list[i]
            assert sep in ("+", "/")
            if sep == "+":
                pass
            else:
                text += "*****"
        assert i == (len(token_list) - 2)
        text += token_list[-1]
        req.append(text.split("*****"))
    return req

@timer()
def parse_en_parse(en_parse):
    assert type(en_parse) == type({}) and "sent" in en_parse
    sent = en_parse["sent"]
    entity_dict = dict(filter(lambda t2: t2[0] != "sent", en_parse.items()))
    return sent, entity_dict

@timer()
def align_entity_rm_part_by_model(en_parse, zh_parse, model, cls, threshold = 50,
    return_two_sim_df = False, batch_threshold = 5000, zh_sent_truncated_len = 35
):
    assert hasattr(cls, "do_one_trans")
    assert type(en_parse) == type(zh_parse) == type({})
    assert "sent" in en_parse
    assert "sent" in zh_parse
    en_sent, en_entity_dict = parse_en_parse(en_parse)
    zh_sent, zh_entity_dict = parse_en_parse(zh_parse)

    if len(zh_sent) > zh_sent_truncated_len:
        zh_sent_truncated = retrieve_all_contain_entity_chips(zh_parse, cp_list)
        assert len(zh_sent_truncated) <= len(zh_sent)
        zh_sent = zh_sent_truncated

        zh_parse = deepcopy(zh_entity_dict)
        zh_parse["sent"] = zh_sent

    assert set(en_entity_dict.keys()) == set(zh_entity_dict.keys())

    en_maintain_entity_tokens = find_max_len_cut_b_with_entity_maintain_j(
'', en_sent,
    reduce(lambda a, b: a + b, en_entity_dict.values()) if en_entity_dict else []
)
    zh_maintain_entity_tokens = find_max_len_cut_b_with_entity_maintain_j(
'', zh_sent,
    reduce(lambda a, b: a + b, zh_entity_dict.values()) if zh_entity_dict else []
)

    #en_join_or_sp_list = produce_join_action(en_maintain_entity_tokens[-1])
    en_rm_entity_token_nest_list = list(map(lambda l:
    list(filter(lambda token:
        all(map(lambda x: x not in token,
                reduce(lambda a, b: a + b, en_entity_dict.values()) if en_entity_dict else []))
        , l))
    , [en_maintain_entity_tokens[1]]))

    zh_join_or_sp_list = produce_join_action(zh_maintain_entity_tokens[-1])

    zh_rm_entity_token_nest_list = list(map(lambda l:
    list(filter(lambda token:
        all(map(lambda x: x not in token,
                reduce(lambda a, b: a + b, zh_entity_dict.values()) if zh_entity_dict else []))
        , l))
    , zh_join_or_sp_list))

    en_part = nest_list_to_ref(en_rm_entity_token_nest_list)
    zh_part = nest_list_to_ref(zh_rm_entity_token_nest_list)

    assert len(en_part) == 1

    #### choose by max
    zh_part_l = list(set(zh_part))
    zh_part_l = list(set(map(lambda x: x.replace(" ", ""), zh_part_l)))
    if len(zh_part_l) <= 1:
        if not zh_part_l:
            zh_sent_rm_entity = ""
        else:
            zh_sent_rm_entity = zh_part_l[0]
    else:
        zh_part_nest_l = batch_as_list(zh_part_l, batch_threshold)

        zh_sent_rm_entity_on_en_req = []
        zh_sent_rm_entity_on_zh_req = []
        for zh_part_l_ in zh_part_nest_l:
            zh_sent_rm_entity_on_en = perm_top_sort_multi_pool(en_part[0],
            json.dumps(zh_part_l_), model, return_score=True)

            zh_sent_rm_entity_on_zh = perm_top_sort_multi_pool(
            cls.do_one_trans(en_part[0])
            , json.dumps(zh_part_l_), model, return_score=True)

            if not zh_sent_rm_entity_on_en_req:
                if not hasattr(zh_sent_rm_entity_on_en, "tolist"):
                    pass
                zh_sent_rm_entity_on_en_req.extend(zh_sent_rm_entity_on_en.tolist())
                zh_sent_rm_entity_on_zh_req.extend(zh_sent_rm_entity_on_zh.tolist())
            else:
                zh_sent_rm_entity_on_en_req.extend(zh_sent_rm_entity_on_en.tolist()[1:])
                zh_sent_rm_entity_on_zh_req.extend(zh_sent_rm_entity_on_zh.tolist()[1:])

        zh_sent_rm_entity_on_en = zh_sent_rm_entity_on_en_req
        zh_sent_rm_entity_on_zh = zh_sent_rm_entity_on_zh_req

        two_sim_df = pd.concat(
        list(map(pd.Series,
        [[en_part[0]] + zh_part_l, zh_sent_rm_entity_on_en, zh_sent_rm_entity_on_zh])),
         axis = 1)
        assert two_sim_df.shape[1] == 3
        assert two_sim_df.shape[0] > 1
        two_sim_df.columns = ["zh_part", "on_en_score", "on_zh_score"]
        two_sim_df["score"] = two_sim_df[["on_en_score", "on_zh_score"]].apply(max, axis = 1)
        two_sim_df = two_sim_df.iloc[1:, :]
        two_sim_df = two_sim_df.sort_values(by = "score", ascending = False)
        if return_two_sim_df:
            return two_sim_df

        zh_sent_rm_entity = two_sim_df["zh_part"].iloc[0]
    #print("zh_sent_rm_entity :", zh_sent_rm_entity)

    rm_maintain_df = pd.DataFrame(list(map(lambda l:
    ("".join(list(filter(lambda token:
        all(map(lambda x: x not in token,
                reduce(lambda a, b: a + b, zh_entity_dict.values()) if zh_entity_dict else []))
        , l))),  list(filter(lambda token:
        any(map(lambda x: x in token,
                reduce(lambda a, b: a + b, zh_entity_dict.values()) if zh_entity_dict else []))
        , l))
    )
    , zh_join_or_sp_list)), columns = ["rm_entity", "maintain_entity"])
    rm_maintain_df["rm_score"] = rm_maintain_df.apply(lambda s: fuzz.ratio(s["rm_entity"], zh_sent_rm_entity),
                                                  axis = 1)
    rm_maintain_df = rm_maintain_df.sort_values(by = "rm_score", ascending=False)
    zh_fix_entity_list = rm_maintain_df.iloc[0]["maintain_entity"]
    return reverse_list_to_dict(zh_entity_dict, zh_fix_entity_list)

@timer()
def guess_should_fix(zh_parse):
    #### can sp_tokens guess???
    zh_sent, zh_entity_dict = parse_en_parse(zh_parse)
    zh_maintain_entity_tokens = find_max_len_cut_b_with_entity_maintain_j(
'', zh_sent,
    reduce(lambda a, b: a + b, zh_entity_dict.values()) if zh_entity_dict else []
)
    #print("zh_maintain_entity_tokens :")
    #print(zh_maintain_entity_tokens)

    req = "Should not Fix"
    for k, v in zh_entity_dict.items():
        assert type(v) == type([])
        for vv in v:
            should_or_not = eval_single_entity(zh_sent, vv)
            if should_or_not == "Should Fix":
                req = "Should Fix"
                break
    return req

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
def drop_duplicates_of_every_df(df):
    if not df.size:
        return df
    ori_columns = df.columns.tolist()
    df["hash"] = df.apply(lambda s: sha512(str(s.to_dict()).encode()).hexdigest(), axis = 1)
    req = []
    k_set = set([])
    for i, r in df.iterrows():
        if r["hash"] not in k_set:
            req.append(r.to_dict())
        k_set.add(r["hash"])
    return pd.DataFrame(req)[ori_columns]

#### model utils
@timer()
def perm_sent_et(**kwargs):
    assert type(kwargs) == type({})
    assert "sent" in kwargs

    sent, entity_dict = parse_en_parse(kwargs)

    entity_list = []
    if entity_dict:
        entity_list = reduce(lambda a, b: a + b , entity_dict.values())
        if entity_list:
            assert type(entity_list[0]) == type("")
    et = list(set(entity_list))
    assert all(map(lambda x: x in sent, et))

    req = []
    for ele_list in permutations(range(len(et))):
        sent_rep = sent
        for i, idx in enumerate(ele_list):
            et_ele = et[i]
            sent_rep = sent_rep.replace(et_ele, "{" + "{}".format(idx) + "}")
        req.append(sent_rep)

    return list(map(lambda x: x.format(*et), req))

@timer()
def perm_top_sort(en_sent ,zh_perm_list, model, return_score = False):
    assert len(zh_perm_list) >= 1
    if len(zh_perm_list) == 1:
        return zh_perm_list[0]
    #### zh_perm_list length too big problem

    embedding = model.encode([en_sent] + zh_perm_list)
    #embedding = embedding_produce_wrapper([en_sent] + zh_perm_list, model, pool = None,)

    sim_m = pytorch_cos_sim(embedding, embedding)
    sim_a = sim_m[0]
    if return_score:
        return sim_a.numpy()
    #### same top val 1
    max_index = np.argsort(sim_a.numpy()[1:])[-1]
    return zh_perm_list[max_index]

@lru_cache(maxsize=52428800)
@timer()
def perm_top_sort_multi_pool(en_sent ,zh_perm_list, model, return_score = False,
    zh_perm_list_len_threshold = 1000
):
    assert type(zh_perm_list) == type("")
    zh_perm_list = json.loads(zh_perm_list)
    if len(zh_perm_list) <= zh_perm_list_len_threshold:
        return perm_top_sort(en_sent ,zh_perm_list, model, return_score = return_score)

    assert hasattr(model, "pool")
    pool = model.pool

    assert len(zh_perm_list) >= 1
    if len(zh_perm_list) == 1:
        return zh_perm_list[0]

    #pool = model.start_multi_process_pool(['cpu'] * 5)

    #### zh_perm_list length too big problem
    #embedding = model.encode([en_sent] + zh_perm_list)
    #embedding = model.encode_multi_process([en_sent] + zh_perm_list, pool, chunk_size=50)
    embedding = embedding_produce_wrapper([en_sent] + zh_perm_list, model, pool = pool,)

    sim_m = pytorch_cos_sim(embedding, embedding)
    sim_a = sim_m[0]
    if return_score:
        return sim_a.numpy()
    #### same top val 1
    max_index = np.argsort(sim_a.numpy()[1:])[-1]
    return zh_perm_list[max_index]


##### may be sent req
##### wrap this aroud model.encode_multi_process and model.encode
##### add flask server with faiss or db

#####
##### request search send post k_list return {"embedding": [list of embedding]}
##### request add send post {"k_list": [], "arr_list": []} no return require.

##### memory error which will kill process in encoding should tackled.
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
def do_one_trans_produce_wrapper(input_sent_list, model, pool = None,
    batch = True, batch_threshold = 1000,
):
    assert hasattr(model, "translate")

    def sent_emb_to_dict(req ,emb, exists_dict):
        if not req:
            assert emb is None
            return exists_dict
        assert type(emb) == type([])
        '''
        if hasattr(emb, "numpy"):
            emb = emb.numpy()
        '''
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
def eval_single_entity(zh_sent, entity_string, sp_tokens = ["的", "?", ".", "。", '', "",
                                                            "是", "什", "么", "吗", "在", "?", "哪",
                                                            "些", "、", "<", ">", "《", "》"
                                                           ],
                    all_or_any = "all"
                                                           ):
    assert all_or_any in ["all", "any"]
    assert entity_string in zh_sent
    left_should = True
    right_should = True

    if zh_sent.startswith(entity_string):
        left_should = False
    if zh_sent.endswith(entity_string):
        right_should = False

    rm_str_list = zh_sent.split(entity_string)
    assert len(rm_str_list) >= 2
    valid_list = []

    for i in range(len(rm_str_list)):
        if i == 0:
            if rm_str_list[i]:
                valid_list.append(rm_str_list[i][-1])
            else:
                valid_list.append('')
        elif i == (len(rm_str_list) - 1):
            if rm_str_list[i]:
                valid_list.append(rm_str_list[i][0])
            else:
                valid_list.append('')
        else:
            sp_part = rm_str_list[i]
            if len(sp_part) <= 1:
                valid_list.append(sp_part)
            elif len(sp_part) >= 2:
                valid_list.append(sp_part[0])
                valid_list.append(sp_part[-1])

    valid_set = set(valid_list)
    #print("valid_set :")
    #print(valid_set)
    #print("-" * 100)
    all_in_sp = True
    for ele in valid_set:
        if ele not in sp_tokens:
            all_in_sp = False
            break
    any_in_sp = False
    for ele in valid_set:
        if ele in sp_tokens:
            any_in_sp = True
            break

    #### left_should, right_should, all_in_sp
    if all_or_any == "all" and all_in_sp:
        return "Should not Fix"
    if all_or_any == "any" and any_in_sp:
        return "Should not Fix"
    return "Should Fix"

@timer()
def use_fix_or_not_by_model(zh_en_entity_dict ,fix_dict, sim_model):
    ####
    # en zh
    ####
    assert type(zh_en_entity_dict) == type({})
    assert len(zh_en_entity_dict) == len(fix_dict)
    assert set(zh_en_entity_dict.keys()) == set(fix_dict.keys())
    req = {}
    for k, v in fix_dict.items():
        assert k in v
        if k == v:
            req[k] = v
        else:
            assert len(v) > len(k)
            #eval_str = eval_single_entity(v, k, all_or_any = "any")
            eval_str = eval_single_entity(v, k, all_or_any = "any", sp_tokens = ["的", "?", ".", "。",
                                                            "是", "什", "么", "吗", "在", "?", "哪",
                                                            "些", "、", "<", ">", "《", "》"
                                                           ])
            assert eval_str in ["Should not Fix", "Should Fix"]
            if eval_str == "Should not Fix":
                req[k] = k
            else:
                #req[k] = v

                zh_entity = k
                assert zh_entity in zh_en_entity_dict
                en_entity = zh_en_entity_dict[zh_entity]
                #req[k] = perm_top_sort_multi_pool(en_entity, list(set([zh_entity, v])), sim_model)
                req[k] = perm_top_sort_multi_pool(en_entity, json.dumps(list(set([zh_entity, v]))), sim_model)
    return req


'''
@timer()
def count_validation(en_parse ,trans_df,
top_1_trans_on_zh, sim_model, cls, entity_cols = [], fix_all = False):
    assert "sent" in trans_df.columns.tolist()
    assert type(entity_cols) == type([])
    assert type(top_1_trans_on_zh) == type("")
    trans_df["sent"] = trans_df["sent"].map(
        lambda zh_sent: re.sub(r"(\{\\fn黑.*\})", "", zh_sent)
    )
    if not fix_all:
        assert top_1_trans_on_zh in trans_df["sent"].tolist()
    else:
        top_1_trans_on_zh = ""
    if not entity_cols:
        entity_cols = list(set(trans_df.columns.tolist()).difference(set(["sent"])))
    #print("entity_cols : ", entity_cols)
    other_cols = list(set(trans_df.columns.tolist()).difference(set(entity_cols + ["sent"])))
    req = trans_df.copy()
    req = drop_duplicates_of_every_df(req)
    req["rm_entity_sent"] = req.apply(lambda s:
                        (reduce(lambda a, b: a + b, s.loc[entity_cols].to_dict().values())
                                           ,find_max_len_cut_b_with_entity_maintain_j(
'', s["sent"],
    reduce(lambda a, b: a + b, s.loc[entity_cols].to_dict().values())
)), axis = 1).map(lambda t2: filter(lambda x: x not in t2[0] ,t2[-1][-1])).map(tuple)

    tuple_cnt_s = req["rm_entity_sent"].value_counts()
    tuple_cnt_min = tuple_cnt_s.min()
    tuple_cnt_max = tuple_cnt_s.max()
    req_cnt = tuple_cnt_min
    #### same rm res may not be error
    if req_cnt > 1:
        pass
    t_list = pd.Series(tuple_cnt_s[tuple_cnt_s >= req_cnt].index.tolist()).map(tuple)
    #print("t_list")
    #print(t_list)
    count_suspect_part = req[
        req["rm_entity_sent"].isin(t_list)
    ][["sent"] + entity_cols + other_cols]


    count_suspect_part["should_fix_or_not"] = count_suspect_part.apply(lambda s: guess_should_fix(
    dict(filter(lambda t2: t2[0] in ["sent"] + entity_cols, s.to_dict().items()))
    ), axis = 1)
    count_suspect_part = count_suspect_part[
        count_suspect_part["should_fix_or_not"] == "Should Fix"
    ]
    #print("count_suspect_part :")
    #print(count_suspect_part)

    if not count_suspect_part.size:
        return
    fix_part = count_suspect_part.iloc[:, :-1]
    assert "sent" in fix_part.columns.tolist()

    if not fix_all:
        fix_part = fix_part[
    fix_part["sent"].isin([top_1_trans_on_zh])
    ]

    if not fix_part.size:
        return

    rp_dict_list = []
    for idx, x in fix_part.iterrows():
        ele = align_entity_rm_part_by_model(en_parse,
        dict(filter(lambda t2: t2[0] in ["sent"] + entity_cols, x.to_dict().items()))
        , sim_model, cls)
        rp_dict_list.append(ele)

    fix_part["rp_dict"] = rp_dict_list

    def produce_zh_en_entity_dict(en_parse, zh_parse):
        assert type(en_parse) == type(zh_parse) == type({})
        assert set(en_parse.keys()) == set(zh_parse.keys())
        req = []
        for k in en_parse.keys():
            if k == "sent":
                continue
            for i, vv in enumerate(en_parse[k]):
                assert type(vv) == type("")
                en_entity = vv
                zh_entity = zh_parse[k][i]
                req.append((zh_entity, en_entity))
        return dict(req)

    fix_part["rp_dict_match"] = fix_part.apply(
        lambda s: use_fix_or_not_by_model(
        produce_zh_en_entity_dict(en_parse, s.loc[["sent"] + entity_cols].to_dict())
        , s["rp_dict"], sim_model), axis = 1
    )
    perm_top_sort_multi_pool.cache_clear()
    return fix_part
'''
def count_validation(en_parse ,trans_df,
top_1_trans_on_zh, sim_model, cls, entity_cols = [], fix_all = False,
                     filter_by_should_fix = True
                     ):
    assert "sent" in trans_df.columns.tolist()
    assert type(entity_cols) == type([])
    assert type(top_1_trans_on_zh) == type("")

    trans_df["sent"] = trans_df["sent"].map(
        lambda zh_sent: re.sub(r"(\{\\fn黑.*\})", "", zh_sent)
    )

    trans_df["sent"] = trans_df["sent"].map(
        lambda zh_sent: re.sub(r"(\{\fn.*\})", "", zh_sent)
    )

    if not fix_all:
        assert top_1_trans_on_zh in trans_df["sent"].tolist()
    else:
        top_1_trans_on_zh = ""
    if not entity_cols:
        entity_cols = list(set(trans_df.columns.tolist()).difference(set(["sent"])))
    print("entity_cols : ", entity_cols)
    other_cols = list(set(trans_df.columns.tolist()).difference(set(entity_cols + ["sent"])))
    req = trans_df.copy()
    req = drop_duplicates_of_every_df(req)
    req["rm_entity_sent"] = req.apply(lambda s:
                        (reduce(lambda a, b: a + b, s.loc[entity_cols].to_dict().values())
                                           ,find_max_len_cut_b_with_entity_maintain_j(
'', s["sent"],
    reduce(lambda a, b: a + b, s.loc[entity_cols].to_dict().values())
)), axis = 1).map(lambda t2: filter(lambda x: x not in t2[0] ,t2[-1][-1])).map(tuple)

    tuple_cnt_s = req["rm_entity_sent"].value_counts()
    tuple_cnt_min = tuple_cnt_s.min()
    tuple_cnt_max = tuple_cnt_s.max()
    req_cnt = tuple_cnt_min
    #### same rm res may not be error
    if req_cnt > 1:
        pass
    t_list = pd.Series(tuple_cnt_s[tuple_cnt_s >= req_cnt].index.tolist()).map(tuple)
    #print("t_list")
    #print(t_list)
    count_suspect_part = req[
        req["rm_entity_sent"].isin(t_list)
    ][["sent"] + entity_cols + other_cols]
    #print("in list :")
    #print(count_suspect_part)
    #print("-" * 100)

    count_suspect_part["should_fix_or_not"] = count_suspect_part.apply(lambda s: guess_should_fix(
    dict(filter(lambda t2: t2[0] in ["sent"] + entity_cols, s.to_dict().items()))
    ), axis = 1)

    #if filter_by_should_fix:
    count_trust_part = count_suspect_part[
        count_suspect_part["should_fix_or_not"] != "Should Fix"
    ].copy()

    if filter_by_should_fix:
        count_suspect_part = count_suspect_part[
        count_suspect_part["should_fix_or_not"] == "Should Fix"
    ]

    #print("count_suspect_part :")
    #print(count_suspect_part)

    def trust_part_to_final_output(req):
        assert "method" in req.columns.tolist()
        if "rp_dict" not in req.columns.tolist():
            req["rp_dict"] = req[
                entity_cols
            ].apply(lambda s: reduce(lambda a, b: a + b, s.tolist()), axis = 1).map(
        lambda l: map(lambda x: (x, x), l)
    ).map(dict)
        if "rp_dict_match" not in req.columns.tolist():
            req["rp_dict_match"] = req[
                entity_cols
            ].apply(lambda s: reduce(lambda a, b: a + b, s.tolist()), axis = 1).map(
        lambda l: map(lambda x: (x, x), l)
    ).map(dict)
        req = req[["sent"] + entity_cols + ["method", "rp_dict", "rp_dict_match"]]
        return req

    if not count_suspect_part.size:
        if count_trust_part.size:
            req = trust_part_to_final_output(count_trust_part)
            return req
        else:
            return

    fix_part = count_suspect_part.iloc[:, :-1]
    assert "sent" in fix_part.columns.tolist()

    if not fix_all:
        fix_part = fix_part[
    fix_part["sent"].isin([top_1_trans_on_zh])
    ]

    if not fix_part.size:
        return

    #print("fix_part :")
    #print(fix_part)

    rp_dict_list = []
    for idx, x in fix_part.iterrows():
        ele = align_entity_rm_part_by_model(en_parse,
        dict(filter(lambda t2: t2[0] in ["sent"] + entity_cols, x.to_dict().items()))
        , sim_model, cls)
        #print("ele in one iter :")
        #print(ele, x)

        rp_dict_list.append(ele)

    fix_part["rp_dict"] = rp_dict_list

    def produce_zh_en_entity_dict(en_parse, zh_parse):
        assert type(en_parse) == type(zh_parse) == type({})
        assert set(en_parse.keys()) == set(zh_parse.keys())
        req = []
        for k in en_parse.keys():
            if k == "sent":
                continue
            for i, vv in enumerate(en_parse[k]):
                assert type(vv) == type("")
                en_entity = vv
                zh_entity = zh_parse[k][i]
                req.append((zh_entity, en_entity))
        return dict(req)

    fix_part["rp_dict_match"] = fix_part.apply(
        lambda s: use_fix_or_not_by_model(
        produce_zh_en_entity_dict(en_parse, s.loc[["sent"] + entity_cols].to_dict())
        , s["rp_dict"], sim_model), axis = 1
    )
    perm_top_sort_multi_pool.cache_clear()

    if count_trust_part.size:
        req = trust_part_to_final_output(count_trust_part)
        assert set(req.columns.tolist()) == set(fix_part.columns.tolist())
        req = pd.concat([fix_part, req], axis = 0)
        assert set(req.columns.tolist()) == set(fix_part.columns.tolist())
        return drop_duplicates_of_every_df(req)

    return fix_part


@timer()
def dump_trans_to_local(trans_obj ,en_parse_list, dump_path, rm_exist = False,
use_methods_input = ("link_on_trans", "link_on_have_trans", "db_et_trans_to_zh_parse_by_dist",
"en_parse_to_zh_parse_by_dist"
),
):
    if os.path.exists(dump_path) and rm_exist:
        shutil.rmtree(dump_path)
    if not os.path.exists(dump_path):
        os.mkdir(dump_path)
    for en_parse in en_parse_list:
        assert type(en_parse) == type({})
        assert "sent" in en_parse

        en_parse_str = str(en_parse)
        file_key = sha512(en_parse_str.encode()).hexdigest()
        file_name = os.path.join(dump_path, "{}.pkl".format(file_key))

        #print("file_name before exists : {}, {} ".format(file_name, os.path.exists(file_name)))

        if os.path.exists(file_name):
            continue

        trans_df = produce_diff_use_method_df(trans_obj, en_parse,
        placeholder_pool, use_methods_input = use_methods_input)

        save_pickle(trans_df, file_name)

@timer()
def dump_edit_to_local(cls, en_parse_list, trans_pkl_path, dump_path, rm_exist = False, pattern = "pkl",
    trans_df_input_cols = ["sent", "e", "t"],
    use_methods = ("link_on_trans", "link_on_have_trans", "db_et_trans_to_zh_parse_by_dist",
    "en_parse_to_zh_parse_by_dist"
    ),
):
    assert hasattr(cls, "do_one_trans")
    assert hasattr(cls, "sim_model")
    if os.path.exists(dump_path) and rm_exist:
        shutil.rmtree(dump_path)
    if not os.path.exists(dump_path):
        os.mkdir(dump_path)

    all_trans_pkl_path = pd.Series(list(Path(trans_pkl_path).glob("*.{}".format(pattern)))).map(
    lambda p: p.__fspath__()
).values.tolist()

    req = pd.DataFrame(pd.Series(
        en_parse_list
    ).map(
        lambda en_parse:
        (en_parse,
        os.path.join(trans_pkl_path, "{}.pkl".format(sha512(str(en_parse).encode()).hexdigest()))
        )
    ).values.tolist(), columns = ["en_parse", "file_path"])
    req = req[
        req["file_path"].map(os.path.exists)
    ]
    if not req.size:
        return
    req["file_path"] = req["file_path"].map(lambda x: x.split("/")[-1])
    req = dict(req[["file_path", "en_parse"]].values.tolist())

    for trans_pkl_path in all_trans_pkl_path:
        p_list = trans_pkl_path.split("/")
        assert len(p_list) >= 2
        req_p_list = [dump_path] + p_list[1:-1] + [p_list[-1]]
        file_name = "/".join(req_p_list)
        assert file_name.endswith(pattern)
        assert len(file_name.split("/")) == len(p_list)

        if os.path.exists(file_name):
            continue

        trans_df = load_pickle(trans_pkl_path)

        assert trans_df is None or hasattr(trans_df, "size")
        if trans_df is None or trans_df.size == 0:
            continue
        #en_parse = trans_df["en_parse"].iloc[0]
        assert file_name.split("/")[-1] in req.keys()
        en_parse = req[file_name.split("/")[-1]]

        if type(en_parse) == type(""):
            en_parse = eval(en_parse)
        assert type(en_parse) == type({})

        trans_df = trans_df[
            trans_df["method"].isin(use_methods)
        ]
        if not trans_df.size:
            continue
        trans_df = trans_df[trans_df_input_cols + ["method"]]


        import pickle as pkl
        with open("last_link_df.pkl", "wb") as f:
            pkl.dump((en_parse ,trans_df), f)
        #print("save !!!!!")
        #return

        count_valid_fix_part = count_validation(en_parse ,trans_df, "", cls.sim_model, cls,
         entity_cols = list(set(trans_df_input_cols).difference(set(["sent"]))), fix_all = True)
        save_pickle(count_valid_fix_part, file_name)

@timer()
def retrieve_not_in_dump(en_parse_list, dump_path):
    continue_times = 0
    for en_parse in en_parse_list:
        assert type(en_parse) == type({})
        assert "sent" in en_parse

        en_parse_str = str(en_parse)
        file_key = sha512(en_parse_str.encode()).hexdigest()
        file_name = os.path.join(dump_path, "{}.pkl".format(file_key))
        if os.path.exists(file_name):
            continue_times += 1
            continue

        return en_parse, continue_times

@timer()
def load_trans_from_local(en_parse_list, dump_path):
    assert os.path.exists(dump_path)
    req = pd.DataFrame(pd.Series(
        en_parse_list
    ).map(
        lambda en_parse:
        (en_parse,
        os.path.join(dump_path, "{}.pkl".format(sha512(str(en_parse).encode()).hexdigest()))
        )
    ).values.tolist(), columns = ["en_parse", "file_path"])
    req = req[
        req["file_path"].map(os.path.exists)
    ]
    final_req = []
    None_en_parse = []
    for idx, r in req.iterrows():
        en_parse, file_path = r["en_parse"], r["file_path"]
        assert os.path.exists(file_path)
        trans_df = load_pickle(file_path)
        assert trans_df is None or hasattr(trans_df, "size")
        if trans_df is None:
            None_en_parse.append(en_parse)
            continue
        trans_df["en_parse"] = str(en_parse)
        trans_df["file_path"] = file_path
        trans_df["en_parse"] = trans_df["en_parse"].map(eval)
        final_req.append(trans_df)
    return pd.concat(final_req, axis = 0), None_en_parse

@timer()
def load_edit_from_local(en_parse_list, trans_pkl_path, dump_path):
    all_trans_df, None_en_parse = load_trans_from_local(en_parse_list, trans_pkl_path)
    assert "file_path" in all_trans_df.columns.tolist()
    if all_trans_df is None or all_trans_df.size == 0:
        return
    assert trans_pkl_path in all_trans_df["file_path"].iloc[0]
    all_trans_df["fix_path"] = all_trans_df["file_path"].map(
        lambda x: x.replace(trans_pkl_path, dump_path)
    )
    all_trans_df = all_trans_df[
        all_trans_df["fix_path"].map(os.path.exists)
    ]
    if all_trans_df is None or all_trans_df.size == 0:
        return

    req = pd.DataFrame(pd.Series(
        en_parse_list
    ).map(
        lambda en_parse:
        (en_parse,
        os.path.join(trans_pkl_path, "{}.pkl".format(sha512(str(en_parse).encode()).hexdigest()))
        )
    ).values.tolist(), columns = ["en_parse", "file_path"])
    req = req[
        req["file_path"].map(os.path.exists)
    ]
    if not req.size:
        return
    req["file_path"] = req["file_path"].map(lambda x: x.split("/")[-1])
    req = dict(req[["file_path", "en_parse"]].values.tolist())

    all_fix_path = all_trans_df["fix_path"].drop_duplicates().tolist()

    final_req = []
    None_en_parse = []
    for file_path in all_fix_path:
        #en_parse, file_path = r["en_parse"], r["file_path"]
        assert file_path.split("/")[-1] in req.keys()
        en_parse = req[file_path.split("/")[-1]]

        assert os.path.exists(file_path)
        trans_df = load_pickle(file_path)
        assert trans_df is None or hasattr(trans_df, "size")
        if trans_df is None:
            None_en_parse.append(en_parse)
            continue
        trans_df["en_parse"] = str(en_parse)
        trans_df["file_path"] = file_path
        trans_df["en_parse"] = trans_df["en_parse"].map(eval)
        final_req.append(trans_df)
    return pd.concat(final_req, axis = 0), None_en_parse

#### log parse func
log_row_pattern = "DEBUG:timer\.(.*):cost (.*)"

def r(path):
    with open(path, "r") as f:
        return f.read()

def parse_log(log_text,
    header = ["func", "time"]):
    all_matched = re.findall(r"{}".format(log_row_pattern), log_text)
    req = pd.DataFrame(all_matched, columns=header)
    #req.iloc[:, 0] = pd.to_datetime(req.iloc[:, 0])
    req["time_in_ms"] = req["time"].map(
    lambda x: float(x.replace(" ms", "")) if x.endswith(" ms") else (
       float(x.replace(" s", "")) * 1000 if x.endswith(" s") else 1 / 0
    )
)
    del req["time"]
    return req

def produce_log_summary(log_path = "trans_link.log"):
    log_text = r(log_path)
    log_df = parse_log(log_text)
    log_func_group_df = log_df.groupby("func")["time_in_ms"].apply(sum).reset_index().sort_values(
    by = "time_in_ms", ascending = False
)
    return log_func_group_df

#### with time func call time expansion test
'''
step_5_log_df = produce_log_summary_with_perc_step("trans_link.log", step_perc = 5)
'''
def produce_log_summary_with_perc_step(log_path = "trans_link.log", step_perc = 10):
    log_text = r(log_path)
    log_df = parse_log(log_text)
    log_df = log_df.reset_index()
    index_perc_array = np.percentile(log_df["index"].values.tolist(), list(range(0, 90, step_perc)),)
    log_df["index_p"] = log_df["index"].map(lambda x: sum(x > index_perc_array))

    req = []

    for step_p, group_df in map(lambda t2: (t2[0] ,t2[1].groupby("func")["time_in_ms"].apply(sum).reset_index().sort_values(
    by = "time_in_ms", ascending = False
)), log_df.groupby("index_p")):
        group_df["step_p"] = step_p
        req.append(group_df)

    return pd.concat(req, axis = 0)

def normalize_df(log_df, normalize_func = "produce_four_trans"):
    assert normalize_func in log_df["func"].values.tolist()
    req = log_df.copy()
    req["time_in_ms"] = req["time_in_ms"] / req[
        req["func"] == normalize_func
    ]["time_in_ms"].iloc[0]
    return req

'''
step_5_log_df = produce_log_summary_with_perc_step("trans_link.log", step_perc = 5)
heavy_time_effect_func_guesser(step_5_log_df, by_func=np.mean)
heavy_time_effect_func_guesser(step_5_log_df, by_func=np.max)

#### default print time expansion line plot
'''
def heavy_time_effect_func_guesser(log_df, pivot_func = "produce_four_trans", threshold = 0.1,
                                  print_heavy_very_with_step = True,
                                  by_func = np.max
                                  ):
    assert "step_p" in log_df.columns.tolist()
    log_df_normalized = pd.concat(list(map(lambda tt2: normalize_df(tt2[1], normalize_func = pivot_func), filter(lambda t2: t2[0] > 0,
                                                    log_df.groupby("step_p")))), axis = 0)
    log_df_normalized_mean_top_some = log_df_normalized.groupby("func")["time_in_ms"].apply(by_func).reset_index().sort_values(by = "time_in_ms", ascending = False)
    log_df_normalized_mean_top_some = log_df_normalized_mean_top_some[
        log_df_normalized_mean_top_some["time_in_ms"] >= threshold
    ]
    if print_heavy_very_with_step:
        tips = log_df_normalized[
    log_df_normalized["func"].isin(log_df_normalized_mean_top_some["func"].values.tolist())
]
        g = sns.FacetGrid(tips, col="func")
        g = g.map(sns.lineplot, "step_p", "time_in_ms")
        return g

    return log_df_normalized_mean_top_some, log_df_normalized

#### log_df cost time fit model (may always use LinearRegression i.e. clf = from sklearn.linear_model import LinearRegression())
'''
step_5_log_df = produce_log_summary_with_perc_step("trans_link.log", step_perc = 5)
X_col, y_col, clf, coef_s, X_y_df = do_fit_on_log_df(step_5_log_df, LinearRegression())

#### show regression coef
pd.DataFrame(coef_s.sort_values(ascending = False)).style.background_gradient(cmap ='viridis')\
        .set_properties(**{'font-size': '10px'})

'''
def do_fit_on_log_df(log_df, clf, y = "produce_four_trans", drop_or_fill = "fill"):
    assert drop_or_fill in ["drop", "fill"]
    assert y in log_df["func"].values.tolist()
    X_y_df = pd.concat(list(map(lambda t2:
            pd.Series(dict(t2[1][["func", "time_in_ms"]].sort_values(by = "func").values.tolist())) ,
                       log_df[
    log_df["step_p"] > 0
].groupby("step_p"))), axis = 1).T
    if drop_or_fill == "drop":
        X_y_df = X_y_df.dropna()
    else:
        X_y_df = X_y_df.fillna(0.0)
    assert y in X_y_df.columns.tolist()
    X = list(filter(lambda x: x != y, X_y_df.columns.tolist()))
    y_col = y
    y = X_y_df[y].values
    X_col = X
    X = X_y_df[X]
    #return X_y_df
    clf.fit(X, y)
    if hasattr(clf, "coef_"):
        coef_s = pd.Series(dict(map(lambda x: (x[1], clf.coef_[x[0]]), enumerate(X_col))))
        return X_col, y_col, clf, coef_s, X_y_df
    return X_col, y_col, clf, X_y_df

####
'''
step_5_log_df = produce_log_summary_with_perc_step("trans_link.log", step_perc = 5)
X_col, y_col, clf, coef_s, X_y_df = do_fit_on_log_df(step_5_log_df, LinearRegression())
produce_corr_imp_df(X_y_df)

produce_corr_imp_df(X_y_df) ->
produce_corr_imp_df(X_y_df, y = "en_parse_to_zh_parse_by_dist").sort_values(by = "corr", ascending = False) ->
produce_corr_imp_df(X_y_df, y = "parse_into_trans_with_produce_aux").sort_values(by = "corr", ascending = False)

### in trans_link.log (use_methods = ("link_on_trans", "en_parse_to_zh_parse_by_dist"))
### produce_four_trans -> en_parse_to_zh_parse_by_dist -> parse_into_trans_with_produce_aux

#### one possibility : iter with php call multi times
#produce_four_trans                   491
#en_parse_to_zh_parse_by_dist         862
#parse_into_trans_with_produce_aux    862
##### average use two times (two php on one trans)
##### normalize "produce_four_trans" time by
##### t3_df["produce_four_trans"] = ((t3_df["produce_four_trans"] / 862) * 491).astype(int)
##### (t3_df.iloc[:, 1] / t3_df.iloc[:, 0]).mean(), (t3_df.iloc[:, 2] / t3_df.iloc[:, 0]).mean()
##### (0.7503296159306718, 0.7457919195441449)
##### i.e. parse_into_trans_with_produce_aux consume 0.75 in produce_four_trans in one php
##### i.e. another possibility : func performance

##### another possibility : func performance

'''
def produce_corr_imp_df(X_y_df, y = "produce_four_trans", corr_threshold = 0.1):
    X_y_corr_df = X_y_df.corr().fillna(0.0)
    X_y_corr_df = X_y_corr_df[
    X_y_corr_df.applymap(lambda x: np.abs(x) > corr_threshold)
]
    corr_s = X_y_corr_df.loc[y].dropna()
    corr_s.name = "corr"
    max_s = X_y_df[X_y_corr_df.loc[y].dropna().index.tolist()].apply(np.max, axis = 0)
    max_s.name = "max"
    return pd.concat([corr_s, max_s], axis = 1).sort_values(by = "max", ascending = False)


##### ner_trans in fix only use do_one_trans sim_model
class NerFix(object):
    @timer()
    def __init__(self, pool_size = 3):
        self.trans_model = EasyNMT('opus-mt')
        self.trans_dict = {}
        self.sim_model = SentenceTransformer('LaBSE')

        t = time()
        pool0 = self.sim_model.start_multi_process_pool(['cpu'] * pool_size)
        print("start time consume :", time() - t)

        self.sim_model.pool = pool0

        t = time()
        pool1 = self.trans_model.start_multi_process_pool(['cpu'] * pool_size)
        print("start time consume :", time() - t)

        self.trans_model.pool = pool1
        ##### model.stop_multi_process_pool(pool)


    @timer()
    @lru_cache(maxsize=52428800)
    def do_one_trans(self, x):
        '''
        l = do_one_trans_produce_wrapper([x], self.trans_model, None)
        assert type(l) == type([])
        assert len(l) == 1
        return l[0]
        '''
        if x in self.trans_dict:
            return self.trans_dict[x]
        trans = self.trans_model.translate([
            x],
                   source_lang="en", target_lang = "zh")[0]
        trans = repeat_to_one(trans)
        self.trans_dict[x] = trans
        return trans
