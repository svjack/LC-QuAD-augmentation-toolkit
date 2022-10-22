#!/usr/bin/env python
# coding: utf-8
#### py377
'''
#### disable log output
import logging, sys
logging.disable(sys.maxsize)
'''

import pandas as pd
import numpy as np
import os
import sqlite_utils
from tqdm import tqdm
from timer import timer

from functools import partial, reduce, lru_cache

from rdflib.graph import Graph
from rdflib_hdt import HDTStore
import json
import os

os.environ["DP_SKIP_NLTK_DOWNLOAD"] = "True"

from deeppavlov import configs
from deeppavlov.core.commands.utils import *
from deeppavlov.core.commands.infer import *
from deeppavlov.core.common.file import *

from deeppavlov.models.kbqa.wiki_parser import *

from scipy.special import softmax

import pandas as pd
import numpy as np
import os
import json

from deeppavlov import configs, build_model
import numpy as np
#pd.set_option('display.max_colwidth', -1)

import inspect

from scipy.special import softmax

from deeppavlov.core.commands.infer import *
from deeppavlov.core.common.file import *

import re
from rapidfuzz import fuzz
from collections import defaultdict
from functools import reduce
from itertools import permutations
from itertools import product

import sys
#sys.path.insert(0, "/Users/svjack/temp/ner_trans")
#sys.path.insert(0, "/temp/ner_trans")

from trans_emb_utils import *
from only_fix_script_ser import *

import difflib
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

import sqlite_utils

'''
sys.path.insert(0, "../kbqa-explore/")
zh_linker_entities = load_pickle("../kbqa-explore/linker_entities.pkl")
'''
#sys.path.insert(0, "kbqa-explore/")
sys.path.insert(0, ".")
zh_linker_entities = load_pickle("kbqa-explore/linker_entities.pkl")


@timer()
def load_data():
    train_df = pd.read_json("train.json")
    test_df = pd.read_json("test.json")
    lcquad_2_0_df = pd.read_json("lcquad_2_0.json")
    df = pd.concat(list(map(lambda x: x[["question", "sparql_wikidata"]], [train_df, test_df, lcquad_2_0_df])), axis = 0).drop_duplicates()
    df = df[["question", "sparql_wikidata"]]
    df = df.dropna().drop_duplicates()
    return df

@timer()
def load_property_info_df(dump_path = "property_info_df.pkl"):
    if os.path.exists(dump_path):
        return load_pickle(dump_path)
    property_info_df = pd.read_json("property_info.json", lines = True)
    property_info_s = property_info_df.apply(lambda x: list(filter(lambda y: not pd.isna(y) ,x.tolist()))[0], axis = 1)
    property_info_df = pd.DataFrame(property_info_s.values.tolist())

    property_info_df["info_dict"] = property_info_df["entities"].map(
    lambda x: info_extracter(x)
)

    property_info_df["en_info"] = property_info_df["info_dict"].map(
        lambda x: x.get("en", [])
    ).map(lambda x: map(clean_single_str, x)).map(list).map(
        lambda l: filter(lambda ele: not desc_matcher(ele), l)
    ).map(list)
    property_info_df["zh_info"] = property_info_df["info_dict"].map(
        lambda x: x.get("zh", [])
    ).map(lambda x: map(clean_single_str, x)).map(list).map(
        lambda l: filter(lambda ele: not desc_matcher(ele), l)
    ).map(list)

    property_info_df["pid"] = property_info_df["entities"].map(
        lambda x: list(x.keys())[0]
    )

    property_info_df = property_info_df[["pid", "en_info", "zh_info"]]
    save_pickle(
        property_info_df, dump_path
    )
    return property_info_df

@timer()
def load_pid_relate_entity_df(generate_dict = True, dump_path = "pid_tuple_on_s_dict.pkl"):
    if generate_dict and os.path.exists(dump_path):
        return load_pickle(dump_path)
    pid_relate_entity_df = pd.read_json("lcquad_pid_relate_entity.json")
    #pid_relate_entity_df = pd.read_json("pid_relate_entity.json")
    pid_relate_entity_df["pid"] = pid_relate_entity_df["l"].map(
        lambda x: x["pid"]
    )
    pid_relate_entity_df["s"] = pid_relate_entity_df["l"].map(
        lambda x: x["s"]
    )
    pid_relate_entity_df = pid_relate_entity_df[["pid", "s"]]

    if generate_dict:
        t2 = pid_relate_entity_df ,produce_pid_tuple_on_s_dict(pid_relate_entity_df)
        save_pickle(
            t2, dump_path
        )
        return t2
    return pid_relate_entity_df

'''
[['"When position did Angela Merkel hold on November 10, 1994?"',
  0,
  'SELECT ?obj WHERE { wd:E1 p:R1 ?s . ?s ps:R1 ?obj . ?s ?p ?x filter(contains(?x, N)) }'],
 ['"What is the boiling point of pressure copper as 4703.0?"',
  1,
  'SELECT ?value WHERE { wd:E1 p:R1 ?s . ?s ps:R1 ?x filter(contains(?x, N)) . ?s ?p ?value }'],
 ['"When did Robert De Niro reside in Marbletown?"',
  2,
  'SELECT ?value WHERE { wd:E1 p:R1 ?s . ?s ps:R1 wd:E2 . ?s ?p ?value }'],
 ['"What are the coordinates for the geographic center of Michigan,
  'as determined by the center of gravity of the surface?"',
  3,
  'SELECT ?obj WHERE { wd:E1 p:R1 ?s . ?s ps:R1 ?obj . ?s ?p wd:E2 }'],
 ['"How many dimensions have a Captain America?"',
  4,
  'SELECT (COUNT(?obj) AS ?value ) { wd:E1 wdt:R1 ?obj }'],
 ['"Which Class IB flammable liquid has the least lower flammable limit?"',
  5,
  'SELECT ?ent WHERE { ?ent wdt:P31 wd:T1 . ?ent wdt:R1 ?obj } ORDER BY ASC(?obj) LIMIT 5'],
 ['"Which member state of the International Centre for Settlement of Investment Disputes has the maximum inflation rate?"',
  6,
  'SELECT ?ent WHERE { ?ent wdt:P31 wd:T1 . ?ent wdt:R1 ?obj . ?ent wdt:R2 wd:E1 } ORDER BY ASC(?obj) LIMIT 5'],
 ['"What periodical literature does Delta Air Lines use as a moutpiece?"',
  7,
  'SELECT ?ent WHERE { wd:E1 wdt:R1 ?ent }']]
'''

wiki_prefix = '''
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX cc: <http://creativecommons.org/ns#>
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX pqn: <http://www.wikidata.org/prop/qualifier/value-normalized/>
PREFIX pqv: <http://www.wikidata.org/prop/qualifier/value/>
PREFIX pr: <http://www.wikidata.org/prop/reference/>
PREFIX prn: <http://www.wikidata.org/prop/reference/value-normalized/>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX prv: <http://www.wikidata.org/prop/reference/value/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdata: <http://www.wikidata.org/wiki/Special:EntityData/>
PREFIX wdno: <http://www.wikidata.org/prop/novalue/>
PREFIX wdref: <http://www.wikidata.org/reference/>
PREFIX wds: <http://www.wikidata.org/entity/statement/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wdtn: <http://www.wikidata.org/prop/direct-normalized/>
PREFIX wdv: <http://www.wikidata.org/value/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
'''


prefix_s = pd.Series(wiki_prefix.split("\n")).map(
    lambda x: x if x.startswith("PREFIX") else np.nan
).dropna().map(
    lambda x: re.findall("PREFIX (.*): <", x)
).map(lambda x: x[0])


prefix_url_dict = dict(map(
    lambda y: (y.split(" ")[1].replace(":", ""), y.split(" ")[2].strip()[1:-1])
    ,filter(
    lambda x: x.strip()
    , wiki_prefix.split("\n"))))

#### entity and property info get by http:
#### not recommand when import only_fix_script_sel SpawnProcess in NerFix pools slow.
re_format = r"{'language': '(.+?)', 'value': '(.+?)'}"
@timer()
def info_extracter(dict_str, lang_filter_list = ["en", "zh"], as_dict = True):
    if type(dict_str) == type({}):
        dict_str = str(dict_str)
    df = pd.DataFrame(re.findall(re_format, dict_str), columns = ["lang", "val"])
    df = df[
        df["lang"].map(
            lambda x: any(map(lambda y: x.lower().startswith(y), lang_filter_list))
        )
    ]
    df["lang"] = df["lang"].map(
        lambda x: x[:2].lower()
    )
    df = df.drop_duplicates()
    df = df.groupby("lang")["val"].apply(list).reset_index()
    if as_dict:
        return dict(df[["lang", "val"]].values.tolist())
    return df

@timer()
def clean_single_str(str_):
    return re.sub("[\(（].*?[\)）]", "", str_)

@timer()
def http_get_wiki_entity_property_info_by_id(id_):
    #### id_, en_info, zh_info = http_get_wiki_entity_property_info_by_id("Q56599233")
    #### id_, en_info, zh_info = http_get_wiki_entity_property_info_by_id("P10")
    assert type(id_) == type("") and (id_.startswith("Q") or id_.startswith("P"))
    url_format = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"
    pid_info_dict = pd.read_json(url_format.format(id_)).to_dict()
    #return pid_info_dict
    dict_str = str(pid_info_dict)
    info_dict = info_extracter(dict_str)
    zh_info = list(map(clean_single_str ,info_dict.get("zh", [])))
    en_info = list(map(clean_single_str ,info_dict.get("en", [])))
    return (id_, en_info, zh_info)

@timer()
def produce_data_dict_for_search(id_seed, dump_path = "info_data_dict.json"):
    if os.path.exists(dump_path):
        with open(dump_path, "r") as f:
            data_dict = json.load(f)
    else:
        data_dict = {}
    assert type(id_seed) == type([])
    assert all(map(lambda x: x.startswith("Q") or x.startswith("P"), id_seed))
    need_id_seed = list(filter(lambda id_: id_ not in data_dict, id_seed))
    for id_ in need_id_seed:
        try:
            id_, en_info, zh_info = http_get_wiki_entity_property_info_by_id(id_)
        except:
            print("err : {}".format(id_))
            continue
        print(id_)
        dump_ele = {
            "en": en_info,
            "zh": zh_info
        }
        data_dict[id_] = dump_ele
        with open(dump_path, "w") as f:
            json.dump(data_dict, f)
    return data_dict

#### cp multi_lang_kb_dict.db produce by ner_trans
#### create table en_zh_so_search as select s, substr(o, 2, length(o) - 2) as o, lang from en_zh_so;
#### create index ss_index on en_zh_so_search (s);
#### create index oo_index on en_zh_so_search (o);
assert os.path.exists("multi_lang_kb_dict.db")
wiki_entity_db = sqlite_utils.Database("multi_lang_kb_dict.db")
assert "en_zh_so_search" in wiki_entity_db.table_names()

@timer()
def retrieve_all_kb_part(query, prefix_s, prefix_url_dict, fullfill_with_url = True):
    req = prefix_s.map(
    lambda x: ("{}:[PQ][0-9]+".format(x))
).map(
    lambda y: re.findall(y, query)
).explode().dropna().drop_duplicates().tolist()
    if fullfill_with_url:
        req = pd.Series(req).map(
            lambda x: prefix_url_dict[x.split(":")[0]] + x.split(":")[1]
        ).tolist()
    return req

@timer()
def retrieve_all_kb_part_wide(query, prefix_s):
    query_tokens = list(filter(lambda x:
    x.strip() and "http" not in x
     ,re.split(r"[\{\} \.]", query)))
    req = set([])
    for token in query_tokens:
        l = prefix_s.map(
    lambda x: ("({}:.+)?".format(x))
).map(
    lambda y: re.findall(y, token)
).explode().dropna().drop_duplicates().tolist()
        for ele in l:
            req.add(ele)
    req = sorted(filter(lambda x: x.strip() ,req))
    return req

@timer()
def find_query_direction_format(entity ,direction = "forw"):
    #### "forw" :: entity format
    #### "backw" :: attribute (prop) format

    assert direction in ["forw", "backw"]
    if direction == "forw":
        query = [f"http://www.wikidata.org/entity/{entity}", "", ""]
    else:
        query = ["", "", f"http://www.wikidata.org/entity/{entity}"]
    return query

@timer()
def one_part_g_producer(one_part_string,
                       format_ = "nt"
                       ):
    from uuid import uuid1
    from rdflib import Graph

    tmp_f_name = "{}.{}".format(uuid1(), format_)
    with open(tmp_f_name, "w") as f:
        f.write(one_part_string)
    g = Graph()
    g.parse(tmp_f_name, format=format_)

    os.remove(tmp_f_name)
    return g

@timer()
def py_dumpNtriple(
    subject, predicate, object_
):
    #### java rdfhdt dumpNtriple python format
    out =[]
    s0 = subject[0]
    if s0=='_' or s0 =='<':
        out.append(subject);
    else:
        out.append('<')
        out.append(subject)
        out.append('>')

    p0 = predicate[0]
    if p0=='<':
        out.append(' ')
        out.append(predicate)
        out.append(' ');
    else:
        out.append(" <")
        out.append(predicate)
        out.append("> ")

    o0 = object_[0]
    if o0=='"':
        #out.append(object_)
        ####
        #UnicodeEscape.escapeString(object.toString(), out);
        #out.append(json.dumps([object_])[1:-1])
        out.append(object_)
        out.append(" .\n");
    elif o0=='_' or o0=='<':
        out.append(object_)
        out.append(" .\n")
    else:
        out.append('<')
        out.append(object_)
        out.append("> .\n")
    return "".join(out)

#@lru_cache(maxsize=52428800)
@timer()
def search_triples_with_parse(source ,query, return_df = True, skip_some_o = True):
    assert hasattr(source, "search_triples")
    iter_, num = source.search_triples(*query)
    req = []
    for s, p, o in iter_:
        o = fix_o(o)
        if skip_some_o:
            if "\n" in o:
                continue
        nt_str = py_dumpNtriple(s, p, o)
        req.append(nt_str)
    g = one_part_g_producer("".join(req))
    if return_df:
        return pd.DataFrame(g.__iter__(), columns = ["s", "p", "o"])
    return g

@timer()
def desc_matcher(x):
    return bool(re.findall(r"(P\d+)", x))

@timer()
def find_query_prop_format(prop, prop_format = "http://www.wikidata.org/prop/direct/{}"):
    query = ["", prop_format.format(prop), ""]
    return query

@timer()
def entity_property_search(prop):
    direct_df, statement_df, prop_df = [None] * 3
    try:
        direct_df = search_triples_with_parse(
    wiki_parser.document,
    find_query_prop_format(prop, "http://www.wikidata.org/prop/direct/{}"), limit = 10240)
    except:
        direct_df = pd.DataFrame(columns = ["s", "p", "o"])
        pass
    try:
        statement_df = search_triples_with_parse(
    wiki_parser.document,
    find_query_prop_format(prop, "http://www.wikidata.org/prop/statement/{}"), limit = 10240)
    except:
        statement_df = pd.DataFrame(columns = ["s", "p", "o"])
    try:
        prop_df = search_triples_with_parse(
    wiki_parser.document,
    find_query_prop_format(prop, "http://www.wikidata.org/prop/{}"), limit = 10240)
    except:
        prop_df = pd.DataFrame(columns = ["s", "p", "o"])

    def row_filter(t3_row):
        s, p, o = t3_row
        s = str(s)
        o_s = str(o)
        s_sp = s.split("/")
        os_sp = o_s.split("/")

        return (len(s_sp) >= 2 and s_sp[-2] == "entity" and "entity/Q" in s and                 len(os_sp) >= 2 and os_sp[-1].startswith("Q"))    or (hasattr(o, "language") and type(o.language) == type("") and "zh" in o.language)

    direct_df, statement_df, prop_df = map(lambda x: x[
        x.apply(row_filter, axis = 1)
    ], [direct_df, statement_df, prop_df])

    all_s_entity = pd.Series(direct_df["s"].tolist() + statement_df["s"].tolist() + prop_df["s"].tolist()).map(
        lambda x: str(x).split("/")[-1] if "entity/Q" in str(x) and str(x).split("/")[-1].startswith("Q") else np.nan
    ).dropna().drop_duplicates()

    all_o_entity = pd.Series(direct_df["o"].tolist() + statement_df["o"].tolist() + prop_df["o"].tolist()).map(
        lambda x: str(x).split("/")[-1] if "entity/Q" in str(x) and str(x).split("/")[-1].startswith("Q") else np.nan
    ).dropna().drop_duplicates()

    ####return all_s_entity
    err_ = []
    all_s_zh = []
    for ele in all_s_entity:
        try:
            t_ele = search_triples_with_parse(
    wiki_parser.document,
    find_query_direction_format(ele, direction="forw"))["o"].map(
            lambda x: str(x) if hasattr(x, "language") and type(x.language) == type("") and "zh" in x.language else np.nan
        ).dropna().drop_duplicates().tolist()
            all_s_zh.append(t_ele)
        except:
            err_.append(ele)

    all_s_zh = pd.Series(all_s_zh).explode().dropna().drop_duplicates()


    all_o_zh = []
    for ele in all_o_entity:
        try:
            t_ele = search_triples_with_parse(
    wiki_parser.document,
    find_query_direction_format(ele, direction="forw"))["o"].map(
            lambda x: str(x) if hasattr(x, "language") and type(x.language) == type("") and "zh" in x.language else np.nan
        ).dropna().drop_duplicates().tolist()
            all_o_zh.append(t_ele)
        except:
            err_.append(ele)

    all_o_zh = pd.Series(all_o_zh).explode().dropna().drop_duplicates()
    return (direct_df, statement_df, prop_df, all_s_zh, all_o_zh, err_)

@timer()
def fix_o(o, rm_char = ["\\"]):
    if not  o.startswith('"'):
        return o
    #print(o)
    assert o.startswith('"')
    num = []
    for i in range(len(o)):
        c = o[i]
        if c == '"':
            num.append(i)
    assert len(num) >= 2
    rm_num = num[1:-1]
    return "".join(
        map(lambda ii: o[ii], filter(lambda i: i not in rm_num and o[i] not in rm_char, range(len(o))))
    )

@timer()
def one_part_g_producer(one_part_string,
                       format_ = "nt"
                       ):
    from uuid import uuid1
    from rdflib import Graph

    tmp_f_name = "{}.{}".format(uuid1(), format_)
    with open(tmp_f_name, "w") as f:
        f.write(one_part_string)
    g = Graph()
    g.parse(tmp_f_name, format=format_)

    os.remove(tmp_f_name)
    return g

'''
wiki_parser = WikiParser(
    wiki_filename = "/Users/svjack/.deeppavlov/downloads/wikidata/wikidata.hdt",
    lang = "",
)
'''
###/Volumes/TOSHIBA EXT/temp/.deeppavlov/downloads/wikidata
'''
wiki_parser = WikiParser(
    wiki_filename = "/Volumes/TOSHIBA EXT/temp/.deeppavlov/downloads/wikidata/wikidata.hdt",
    lang = "",
)
'''
wiki_parser = WikiParser(
    wiki_filename = "kbqa-explore/wikidata.hdt",
    lang = "",
)


re_format = r"{'language': '(.+?)', 'value': '(.+?)'}"
@timer()
def info_extracter(dict_str, lang_filter_list = ["en", "zh"], as_dict = True):
    if type(dict_str) == type({}):
        dict_str = str(dict_str)
    df = pd.DataFrame(re.findall(re_format, dict_str), columns = ["lang", "val"])
    df = df[
        df["lang"].map(
            lambda x: any(map(lambda y: x.lower().startswith(y), lang_filter_list))
        )
    ]
    df["lang"] = df["lang"].map(
        lambda x: x[:2].lower()
    )
    df = df.drop_duplicates()
    df = df.groupby("lang")["val"].apply(list).reset_index()
    if as_dict:
        return dict(df[["lang", "val"]].values.tolist())
    return df

@timer()
def clean_single_str(str_):
    return re.sub("[\(（].*?[\)）]", "", str_)

@timer()
def merge_nest_list(nest_list, threshold = 0):
    req = []
    for ele in nest_list:
        if not req:
            req.extend(ele)
        else:
            b, e = ele
            if b - req[-1] <= threshold:
                req.append(e)
    return [req[0], req[-1]]

@timer()
def get_match_blk_by_diff(a, b, filter_size = 0,
                          threshold = 0
                         ):
    s = difflib.SequenceMatcher(lambda x: None,
                  a,
                    b)
    m_blocks = list(filter(lambda m: m.size > filter_size,
                           s.get_matching_blocks()))
    def m_block_to_b(m_blocks):
        b_part = list(map(lambda m: [m.b, m.b + m.size], m_blocks))
        return b_part
    b_part = m_block_to_b(m_blocks)

    if not b_part:
        return ""
    bb, ee = merge_nest_list(b_part, threshold=threshold)

    return b[bb:ee]

@timer()
def get_match_blk(a, b, threshold_bnd = [0, 10], match_first_end = True):
    '''
    get_match_blk("成交楼面均价", "平均溢价率小于10的成交楼面平均价格是多少")
    '''
    req_list = []
    for thre in range(threshold_bnd[0], threshold_bnd[-1]):
        ret = get_match_blk_by_diff(a, b, threshold=thre)
        #print("ret :", ret)
        if ret and ret in b:
            req_list.append(ret)

        if ret:
            if match_first_end:
                if ret.startswith(a[0]) and ret.endswith(a[-1]):
                    req_list.append(ret)
            else:
                req_list.append(ret)

    #print("req_list :", req_list)
    if req_list:
        first_ele =        sorted(req_list, key=lambda x: -1 * len(x))[0]
        if fuzz.ratio(first_ele.lower(), a.lower()) >= 50:
            return first_ele
        ele_a, ele_a_score = cut_compare(first_ele.lower(), a.lower())
        if ele_a_score > 0.7:
            return first_ele
    return None

@timer()
def get_match_intersection(a, b, threshold_bnd = [0, 5], match_first_end = True):
    '''
    get_match_blk("成交楼面均价", "平均溢价率小于10的成交楼面平均价格是多少")
    '''
    req_list = []
    for thre in range(threshold_bnd[0], threshold_bnd[-1]):
        ret = get_match_blk_by_diff(a, b, threshold=thre)
        #print("ret :", ret)
        if ret and ret in b:
            req_list.append(ret)

        if ret:
            if match_first_end:
                if ret.startswith(a[0]) and ret.endswith(a[-1]):
                    req_list.append(ret)
            else:
                req_list.append(ret)

    #print("req_list :", req_list)
    if req_list:
        first_ele = sorted(req_list, key=lambda x: -1 * len(x))[0]
        return first_ele
    return None

@timer()
def get_match_intersection_fill_gap(a, b, c,
    threshold_bnd = [0, 5],
    match_first_end = True,
                                   ):
    x = get_match_intersection(a, b, threshold_bnd = threshold_bnd, match_first_end = match_first_end)
    return x
    if type(x) != type("") or len(x.strip()) == 0:
        return x
    assert type(x) == type("")
    if x not in c:
        return x
    def process_token(token, rm_char_list = ["?"]):
        for rm_c in rm_char_list:
            token = token.replace(rm_c, "")
        return token
    c_x_sp = c.split(x)
    assert len(c_x_sp) >= 2

    c_x_sp_ = []
    for idx in range(len(c_x_sp)):
        if idx in [0, len(c_x_sp) - 1]:
            c_x_sp_.append(c_x_sp[idx])
        else:
            xx = c_x_sp[idx]
            xx = x if not xx else xx
            c_x_sp_.append(xx)
    c_x_sp = c_x_sp_

    c_x_sp_l = pd.Series(unzip_string(c_x_sp)).map(
        lambda t2: (
            t2[0].split(" ")[-1] if t2[0].strip() else t2[0],
            t2[1].split(" ")[0] if t2[1].strip() else t2[1]
        )
    ).map(
        lambda tt2: (process_token(tt2[0]), process_token(tt2[1]))
    ).map(
        lambda ttt2: ttt2[0] + x + ttt2[1]
    ).map(
        lambda y: y.strip()
    ).tolist()
    assert c_x_sp_l
    req = sorted(c_x_sp_l, key = len)[0]
    assert req in c
    assert phrase_validation(c, req)
    return req

@timer()
def generate_score_percentile_slice(score_df, score_col = "fuzz", threshold = 5):
    max_score = score_df[score_col].max()
    max_ratio = int((score_df[score_df[score_col] == max_score].shape[0] / score_df.shape[0]) * 100)
    threshold = max(threshold, max_ratio)
    score_p = np.percentile(score_df[score_col].values, 100 - threshold)
    return score_df[
        score_df[score_col] >= score_p
    ]

@timer()
def sent_list_match_to_df(en_sent, l, slice_by_perc_head = True):
    assert type(en_sent) == type("")
    assert type(l) == type([])
    assert bool(l)

    df = pd.DataFrame(
pd.Series(l).map(
    lambda x: (x ,get_match_intersection_fill_gap(en_sent, x, en_sent))
).values.tolist()
)
    df.columns = ["string", "inter_str"]
    df = df[
        df.apply(lambda x: None not in x.tolist() and
        all(map(lambda y: bool(y.strip()), x))
        , axis = 1)
    ]
    if not df.size:
        return None
    df["score"] = df.apply(lambda s: fuzz.ratio(*s.tolist()), axis = 1)
    req = df.sort_values(by = "score", ascending = False)
    if slice_by_perc_head:
        return generate_score_percentile_slice(req, "score", 1)
    return req

@timer()
def sent_list_match_to_df_with_bnd(en_sent, l, slice_by_perc_head = True, upper_bnd = 5):
    assert upper_bnd > 0
    assert type(en_sent) == type("")
    assert type(l) == type([])
    assert bool(l)

    df = pd.DataFrame(
pd.Series(l).map(
lambda x: (x ,get_match_intersection_fill_gap(en_sent, x, en_sent, threshold_bnd = [0, upper_bnd]))
).values.tolist()
)
    df.columns = ["string", "inter_str"]
    df = df[
        df.apply(lambda x: None not in x.tolist() and
        all(map(lambda y: bool(y.strip()), x))
        , axis = 1)
    ]
    if not df.size:
        return None

    df["score"] = df.apply(lambda s: fuzz.ratio(*s.tolist())
                                     , axis = 1)

    req = df.sort_values(by = "score", ascending = False)
    if slice_by_perc_head:
        return generate_score_percentile_slice(req, "score", 1)
    return req

@timer()
def sent_list_match_to_df_bnd_cat(
    en_sent, l, slice_by_perc_head = True, upper_bnd = 5,
    length_ratio_threshold = 0.5
):
    assert upper_bnd > 0
    req = []
    for i in range(1, upper_bnd):
        df = sent_list_match_to_df_with_bnd(
    en_sent,
                      l
                      , False, i
)
        if not hasattr(df, "size") or df.size == 0:
            continue
        df["bnd"] = i

        df["in_sent"] = df["inter_str"].map(lambda x: x in en_sent)
        df["is_phrase"] = df["inter_str"].map(
        lambda x: phrase_validation(en_sent, x.strip()) if (en_sent.strip() and x.strip()) else False
    )
        req.append(df)
    if not req:
        return None

    df = pd.concat(req, axis = 0).sort_values(by = ["is_phrase", "in_sent", "score"],
                                              ascending = False)
    df = drop_duplicates_by_col(df, "string")[["string", "inter_str", "score"]]
    df["fuzz"] = df["score"]
    df["score"] = df.shape[0] - np.arange(df.shape[0])
    df_tmp = df.copy()
    df_tmp["inter_str_ratio"] = df_tmp.apply(lambda x: len(x["inter_str"]) / len(x["string"]), axis = 1)
    df_tmp = df_tmp[
        df_tmp["inter_str_ratio"] >= length_ratio_threshold
    ]
    if df_tmp.size > 0:
        df = df_tmp
    else:
        pass

    df = df[["string", "inter_str", "score"]]
    req = df
    if slice_by_perc_head:
        return generate_score_percentile_slice(req, "score", 1)
    return df

@timer()
def lemmatize_one_token(token, wn = WordNetLemmatizer(),
                        all_stem_keys = list(wordnet.MORPHOLOGICAL_SUBSTITUTIONS.keys()),
                       major_key = "v"
                       ):
    assert type(token) == type("")
    assert major_key in all_stem_keys
    major_token = wn.lemmatize(token, major_key)
    if major_token != token:
        return major_token
    return sorted(map(lambda k: wn.lemmatize(token, k),
                      set(all_stem_keys).difference(
                          set([major_key])
                      )
                     ),
                 key = len
                 )[0]

@timer()
def lemma_score_match_it(en_sent, en_info, wn = WordNetLemmatizer(),
                        slice_by_perc_head = True
                        ):
    sent_trans =  " ".join(list(filter(lambda x: x.strip() ,
                              map(lambda y:
                              lemmatize_one_token(process_token(y))
                               ,en_sent.split(" "))
                             )
                      ))
    sent_info_match_df = sent_list_match_to_df_bnd_cat(en_sent, en_info, False)
    #sent_info_match_df = sent_list_match_to_df(en_sent, en_info, False)
    if sent_info_match_df is None or not sent_info_match_df.size:
        return None

    sent_info_match_df["is_in"] = sent_info_match_df.apply(
    lambda s:
        lemmatize_one_token(sorted(filter(lambda x: x.strip() ,
               s["inter_str"].strip().split(" ")), key = len, reverse = True)[0]) in sent_trans if\
               sorted(filter(lambda x: x.strip() ,
                      s["inter_str"].strip().split(" ")), key = len, reverse = True) else False
                               , axis = 1
)
    sent_info_match_df["is_phrase"] = sent_info_match_df.apply(
        lambda s: phrase_validation(s["string"].strip(), s["inter_str"].strip()), axis = 1
    )
    sent_info_match_df = sent_info_match_df.sort_values(
        by = ["is_phrase", "is_in", "score"], ascending = False
    )
    if not sent_info_match_df.size:
        return None
    if slice_by_perc_head:
        return sent_info_match_df.head(1)
    return sent_info_match_df

@timer()
def guess_sim_representation(en_sent, entity_prop_dict, retrieve_top1 = True,
    use_lemma = True
):
    assert type(en_sent) == type("")
    assert type(entity_prop_dict) == type({})
    req_entity_prop_dict = {}
    for k, v in entity_prop_dict.items():
        assert type(v) == type([])
        if use_lemma:
            req_entity_prop_dict[k] = lemma_score_match_it(en_sent, v, slice_by_perc_head = True)
        else:
            req_entity_prop_dict[k] = sent_list_match_to_df_bnd_cat(en_sent, v, True)
            #req_entity_prop_dict[k] = sent_list_match_to_df(en_sent, v, True)

    req = {}
    for k in req_entity_prop_dict.keys():
        if req_entity_prop_dict[k] is None:
            continue
        else:
            assert hasattr(req_entity_prop_dict[k], "size")
            assert req_entity_prop_dict[k].size > 0
            req_entity_prop_dict[k]["fuzz"] = req_entity_prop_dict[k]["string"].map(
                lambda x: fuzz.ratio(en_sent, x)
            )
            req_entity_prop_dict[k]["fuzz_score"] = req_entity_prop_dict[k].apply(
                lambda s: s["fuzz"] * s["score"]
                , axis = 1)
            req_entity_prop_dict[k] = req_entity_prop_dict[k].sort_values(
                by = "fuzz_score", ascending = False
            )
            if retrieve_top1:
                req_entity_prop_dict[k] = req_entity_prop_dict[k][["string", "inter_str"]].iloc[0].tolist()
            req[k] = req_entity_prop_dict[k]
    return req
    #return req_entity_prop_dict

@timer()
def guess_sim_representation_by_score(en_sent, entity_prop_dict, agg_func = sum):
    use_lemma_dict = guess_sim_representation(en_sent, entity_prop_dict, use_lemma = True)
    ori_dict = guess_sim_representation(en_sent, entity_prop_dict, use_lemma = False)
    if not use_lemma_dict:
        if ori_dict:
            return ori_dict
    if not ori_dict:
        if use_lemma_dict:
            return use_lemma_dict
    if not ori_dict and not use_lemma_dict:
        return {}

    def process_token(token, rm_char_list = ["?"]):
        for rm_c in rm_char_list:
            token = token.replace(rm_c, "")
        return token

    sent_tokens = list(filter(lambda x: lemmatize_one_token(x.strip().lower()) ,
                              map(lambda y: process_token(y) ,en_sent.split(" "))
                             )
                      )

    #print(sent_tokens)

    def dict_to_set(dict_):
        dict_s = set(reduce(lambda a, b:a + b ,dict_.values()))
        dict_s = set(map(lambda y: y.lower() ,
        reduce(lambda a, b : a + b ,map(lambda x: x.split(" "),
        dict_s))))
        dict_s = set(map(lemmatize_one_token, dict_s))
        return dict_s

    use_lemma_score = agg_func(map(lambda y: fuzz.ratio(y[0], y[1])
     ,filter(lambda x: len(x) == 2 ,use_lemma_dict.values())))
    use_lemma_s = dict_to_set(use_lemma_dict)
    #use_lemma_cnt = len(list(filter(lambda x: x in use_lemma_s, sent_tokens)))
    #use_lemma_cnt = max(map(lambda y: max(map(lambda z: fuzz.ratio(y, z), use_lemma_s)), sent_tokens))
    use_lemma_cnt = sum(map(lambda y: max(map(lambda z: fuzz.ratio(y, z), use_lemma_s)), sent_tokens))
    use_lemma_score = use_lemma_cnt * use_lemma_score
    #print(use_lemma_dict, use_lemma_s ,use_lemma_cnt, use_lemma_score)

    ori_score = agg_func(map(lambda y: fuzz.ratio(y[0], y[1])
     ,filter(lambda x: len(x) == 2 ,ori_dict.values())))
    ori_s = dict_to_set(ori_dict)
    #ori_cnt = len(list(filter(lambda x: x in ori_s, sent_tokens)))
    #ori_cnt = max(map(lambda y: max(map(lambda z: fuzz.ratio(y, z), ori_s)), sent_tokens))
    ori_cnt = sum(map(lambda y: max(map(lambda z: fuzz.ratio(y, z), ori_s)), sent_tokens))

    ori_score = ori_cnt * ori_score
    #print(ori_dict, ori_s ,ori_cnt, ori_score)

    return use_lemma_dict if use_lemma_score > ori_score else ori_dict

@timer()
def map_reduce_guess_sim_representation_by_score(en_sent, entity_prop_dict):
    dict__ = {}
    entity_prop_dict = deepcopy(entity_prop_dict)
    entity_prop_dict_ = {}
    for k, v in entity_prop_dict.items():
        assert type(v) == type([])
        v_in = list(filter(lambda x: x in en_sent, v))
        if v_in:
            v_in_ele = sorted(v_in, key = len, reverse = True)[0]
            assert type(v_in_ele) == type("")
            dict__[k] = [v_in_ele, v_in_ele]
        else:
            entity_prop_dict_[k] = v
    entity_prop_dict = entity_prop_dict_

    if entity_prop_dict:
        req = {}
        for k, v in entity_prop_dict.items():
            assert type(v) == type([])
            req[k] = list(map(lambda x: x.lower(), v))

        dict_lower = dict(reduce(lambda a, b: a + b ,map(lambda x:
        list(guess_sim_representation_by_score(en_sent ,dict([x])).items())
         ,req.items())))

        #print("dict_lower :")
        #print(dict_lower)

        dict_ = dict(reduce(lambda a, b: a + b ,map(lambda x:
        list(guess_sim_representation_by_score(en_sent ,dict([x])).items())
         ,entity_prop_dict.items())))

        #print("dict_ :")
        #print(dict_)

        all_keys_set = set(dict_lower.keys()).union(set(dict_.keys()))
        dict_emp = {}
        dict_lower_emp = {}
        for k in all_keys_set:
            v_ = dict_.get(k, None)
            v_lower = dict_lower.get(k, None)

            if v_ is not None:
                dict_emp[k] = v_
            else:
                assert v_lower is not None
                dict_emp[k] = v_lower

            if v_lower is not None:
                dict_lower_emp[k] = v_lower
            else:
                assert v_ is not None
                dict_lower_emp[k] = v_

        dict_ = dict_emp
        dict_lower = dict_lower_emp

        assert set(dict_lower.keys()) == set(dict_.keys())
        #dict__ = {}
        for k in dict_.keys():
            v_ = dict_[k]
            v_lower = dict_lower[k]
            assert type(v_) == type([])
            assert type(v_lower) == type([])

            v_score = sum(map(lambda x: fuzz.ratio(x, en_sent), v_))
            v_lower_score = sum(map(lambda x: fuzz.ratio(x, en_sent), v_lower))

            v__ = v_lower if v_lower_score > v_score else v_
            dict__[k] = v__

    return dict__


##### some entityid not in now version kb
#@lru_cache(maxsize=52428800)
@timer()
def search_entity_rep_by_lang_filter(entityid, lang = "en"):
    forw_df = search_triples_with_parse(
    wiki_parser.document,
    find_query_direction_format(entityid, direction="forw"))
    assert hasattr(forw_df, "size")
    if not forw_df.size:
        return []
    req = forw_df[
        forw_df["o"].map(
            lambda x: hasattr(x, "language") and type(x.language) == type("") and x.language.startswith(lang)
        )
    ]["o"].map(str).drop_duplicates().tolist()
    return req

@timer()
def search_entity_rep_by_lang_filter_in_db(entityid, wiki_entity_db, lang = "en"):
    id = entityid
    g = wiki_entity_db.query("select * from en_zh_so_search where s = '{}' and lang = '{}'".format(id, lang))
    l = list(g)
    if not l:
        return []
    df = pd.DataFrame(l)
    l = df["o"].drop_duplicates().tolist()
    return l

#### only read content from produce_data_dict_for_search
data_dict = produce_data_dict_for_search([])
#### full fill only for
#### "find_zh_str_entityid_by_linking"
#### not
#### "search_sim_entity_by_property_count_by_dict_add_fuzz"
full_fill_empty_zh_info_by_en_info = True
if full_fill_empty_zh_info_by_en_info:
    data_dict = dict(map(lambda t2: (
        t2[0], {
            "en": t2[1]["en"],
            "zh": t2[1]["zh"] if t2[1]["zh"] else t2[1]["en"]
        }
    ),
    data_dict.items()))
@timer()
def search_entity_rep_by_lang_filter_by_init_dict(entityid, lang = "en", data_dict = data_dict):
    assert type(data_dict) == type({})
    if entityid in data_dict:
        if lang in data_dict[entityid]:
            return data_dict[entityid][lang]
    return search_entity_rep_by_lang_filter(entityid, lang)

@timer()
def find_zh_str_entityid_by_linking(zh_str, zh_linker_entities,
    fuzz_threshold = 60, lower_threshold = 50, wiki_entity_db = wiki_entity_db
):
    assert type(zh_str) == type("")
    assert callable(zh_linker_entities)
    a, b = zh_linker_entities([[zh_str]])
    entity_id_list = np.asarray(a).reshape([-1]).tolist()
    collect_list = []
    for entityid in entity_id_list:
        #zh_list = search_entity_rep_by_lang_filter(entityid, "zh")
        #zh_list = search_entity_rep_by_lang_filter_by_init_dict(entityid, "zh")
        zh_list = search_entity_rep_by_lang_filter_in_db(entityid, wiki_entity_db, "zh")
        if not zh_list:
            zh_list = search_entity_rep_by_lang_filter_by_init_dict(entityid, "zh")

        if not zh_list:
            continue
        score = max(map(lambda x: fuzz.ratio(x, zh_str), zh_list))
        if score >= fuzz_threshold:
            return (entityid, zh_list)
        else:
            if score >= lower_threshold:
                collect_list.append((entityid, zh_list, score))
    if collect_list:
        return sorted(collect_list, key = lambda t3: t3[-1], reverse = True)[0][:2]
    return None

@timer()
def clean_str_into_db(str_):
    #print(str_)
    return str_.replace(",", "").replace("'", "").replace('"', "").replace(".", "").replace("-", "").\
    replace('(', '').replace(')', '').replace("/", "").replace("?", "").replace("+", "").\
    replace(":", "").replace("&", "").replace("~", "").replace("*", "")

@timer()
def find_zh_str_entityid_by_db(zh_str, wiki_entity_db, lang = "zh"):
    zh_str = clean_str_into_db(zh_str)
    g = wiki_entity_db.query("select * from en_zh_so_search where o = '{}'".format(zh_str))
    l = list(g)
    if not l:
        return None
    df = pd.DataFrame(l)
    df = df[
        df["lang"] == lang
    ]
    if not df.size:
        return None
    id = df["s"].iloc[0]
    g = wiki_entity_db.query("select * from en_zh_so_search where s = '{}' and lang = '{}'".format(id, lang))
    l = list(g)
    if not l:
        return None
    df = pd.DataFrame(l)
    l = df["o"].drop_duplicates().tolist()
    return (id, l)

#### some entityid not have correspond zh desc, use en desc find a near zh desc and find its entityid
#### use NerFix in only_fix_script_sel.py
@timer()
def find_en_str_entityid_by_trans_near_linking(en_str, zh_linker_entities, ner_fix,
    fuzz_threshold = 80, zh_flag_filter_l = ["n", "v"], need_size = 3, show_score_df = False):
    assert type(en_str) == type("")
    assert hasattr(ner_fix, "do_one_trans")
    zh = ner_fix.do_one_trans(en_str)
    assert type(zh) == type("")
    zh_l = sorted(map(lambda tt2: tt2[0] ,filter(lambda t2: any(map(lambda y: t2[1].startswith(y), zh_flag_filter_l))
                         ,map(lambda x: (x.word, x.flag) ,posseg.lcut(zh)))),
                        key = lambda xx: fuzz.ratio(xx, zh), reverse = True
                        )
    if not zh_l:
        zh_l = [zh]
    #score_arr = perm_top_sort(en_str, zh_l, ner_fix.sim_model, return_score = True)
    score_arr = perm_top_sort(zh, zh_l, ner_fix.sim_model, return_score = True)
    if type(score_arr) == type(""):
        #### only one not compare
        #### maintain zh_l
        assert len(zh_l) == 1
    else:
        zh_score_df = pd.concat(
        [pd.Series(zh_l), pd.Series(score_arr.reshape([-1])[1:].tolist())],
        axis = 1
    )
        zh_score_df.columns = ["zh", "score"]
        if show_score_df:
            print(zh_score_df)
        zh_l = zh_score_df.sort_values(by = "score", ascending = False)["zh"].tolist()

    zh_l = [zh] + zh_l

    zh_ll = []
    for ele in zh_l:
        if ele not in zh_ll and len(zh_ll) < need_size:
            zh_ll.append(ele)

    zh_id_l = list(filter(lambda t2: t2[1] is not None ,map(
        lambda zh_str: (zh_str,
                       find_zh_str_entityid_by_linking(zh_str, zh_linker_entities, fuzz_threshold = fuzz_threshold)
                       )
        , zh_ll)))

    zh_id_df = pd.DataFrame(zh_id_l)
    return zh_id_df

##### time consume to produce a pid_tuple with zh_entity dict
@timer()
def produce_pid_tuple_on_s_dict(pid_relate_entity_df):
    s_on_pid_df = pid_relate_entity_df.explode("s").groupby("s")["pid"].apply(set).map(sorted).map(tuple).reset_index()
    pid_tuple_on_s_dict = dict(s_on_pid_df[["pid", "s"]].groupby("pid")["s"].apply(set).map(list).map(tuple).reset_index().values.tolist())
    return pid_tuple_on_s_dict

@timer()
def load_pid_tuple_on_s_dict_zh_entity_search_table(pid_tuple_on_s_dict
    ,dump_path = "pid_tuple_on_s_dict.db"):
    assert type(pid_tuple_on_s_dict) == type({})
    if os.path.exists(dump_path):
        return sqlite_utils.Database(dump_path)
    db = sqlite_utils.Database(dump_path)
    s = pd.DataFrame(pid_tuple_on_s_dict.items())[1].explode().drop_duplicates().dropna().map(
    lambda x: (" ".join(jieba.lcut(x)), x)
)
    db["zh_entity"].insert_all(
    s.map(lambda t2: {"string": t2[1], "fts_string": t2[0]}).tolist()
)
    db["zh_entity"].enable_fts(["fts_string"])
    return db

@timer()
def search_sim_entity_by_property_count_by_dict(entityid,
pid_tuple_on_s_dict):
    assert type(entityid) == type("")
    assert type(pid_tuple_on_s_dict) == type({})

    forw_df = search_triples_with_parse(
    wiki_parser.document,
    find_query_direction_format(entityid, direction="forw"))

    entity_related_properties = forw_df["p"].map(
    lambda x: x if str(x).startswith("http://www.wikidata.org/prop") else np.nan
).dropna().map(
    lambda x: str(x).split("/")[-1]
).drop_duplicates().tolist()

    assert type(entity_related_properties) == type([])
    entity_related_properties = set(entity_related_properties)

    req = defaultdict(list)
    for key_tuple_t2 in sorted(pid_tuple_on_s_dict.items(), key = lambda t2: len(t2[0]), reverse = True):
        assert type(key_tuple_t2) == type((1,))
        key_tuple, string_tuple = key_tuple_t2
        inter_nun = len(set(key_tuple).intersection(entity_related_properties))
        assert type(string_tuple) == type((1,))
        req[inter_nun].extend(
            list(string_tuple)
        )

    cnt_df = pd.DataFrame(
        req.items()
    , columns = ["pid", "s"])
    cnt_df = cnt_df.sort_values(by = "pid", ascending = False)[["s", "pid"]].head(100)
    cnt_df = cnt_df.explode("s").sort_values(by = "pid", ascending = False)
    return cnt_df

@timer()
def search_sim_entity_by_property_count_by_dict_add_fuzz(entityid,
pid_tuple_on_s_dict, ner_fix, wiki_entity_db = wiki_entity_db):
    cnt_df = search_sim_entity_by_property_count_by_dict(entityid, pid_tuple_on_s_dict)
    zh_first = None

    zh = search_entity_rep_by_lang_filter_in_db(entityid, wiki_entity_db, "zh")
    if not zh:
        zh = search_entity_rep_by_lang_filter_by_init_dict(entityid, "zh")
    en = search_entity_rep_by_lang_filter_in_db(entityid, wiki_entity_db, "en")
    if not en:
        en = search_entity_rep_by_lang_filter_by_init_dict(entityid, "en")

    if zh != en:
        zh_first = zh[0]
    else:
        assert type(en) == type([])
        if not en:
            return cnt_df
        en_first = en[0]
        zh_first = ner_fix.do_one_trans(en_first)
    assert type(zh_first) == type("")
    cnt_df["pid"] = cnt_df.apply(
        lambda s: fuzz.ratio(s["s"], zh_first) * s["pid"]
    , axis = 1)
    cnt_df = cnt_df.sort_values(
        by = "pid", ascending = False
    )
    return cnt_df

@timer()
def search_sim_entity_by_property_count_by_dict_add_fuzz_f_by_db(entityid,
pid_tuple_on_s_dict, ner_fix, zh_entity_db, slice_size = 10000):
    cnt_df = search_sim_entity_by_property_count_by_dict(entityid, pid_tuple_on_s_dict)
    zh_first = None

    zh = search_entity_rep_by_lang_filter_in_db(entityid, wiki_entity_db, "zh")
    if not zh:
        zh = search_entity_rep_by_lang_filter_by_init_dict(entityid, "zh")
    en = search_entity_rep_by_lang_filter_in_db(entityid, wiki_entity_db, "en")
    if not en:
        en = search_entity_rep_by_lang_filter_by_init_dict(entityid, "en")

    if zh != en and zh:
        zh_first = zh[0]
    else:
        assert type(en) == type([])
        if not en:
            return cnt_df
        en_first = en[0]
        zh_first = ner_fix.do_one_trans(en_first)
    assert type(zh_first) == type("")
    zh_first_cut = set(
    filter(lambda x: x.strip() ,jieba.lcut(zh_first))
    )
    assert zh_first_cut
    s = pd.Series(
        list(zh_first_cut)
    ).map(clean_str_into_db).map(lambda x: x if x.strip() else np.nan).dropna().drop_duplicates()\
    .map(
        lambda x: list(zh_entity_db["zh_entity"].search(x))
    ).explode().dropna().map(str).drop_duplicates().map(eval)
    if s.size:
        need_entities = s.map(
            lambda x: x["string"]
        ).dropna().drop_duplicates().tolist()
    else:
        need_entities = []
    assert "s" in cnt_df.columns.tolist()
    if need_entities:
        cnt_df_tmp = cnt_df[
            cnt_df["s"].isin(need_entities)
        ]
        if cnt_df_tmp.size:
            cnt_df = cnt_df_tmp
        else:
            cnt_df_tmp = cnt_df[
                 cnt_df["pid"] > 0
            ]
            if cnt_df_tmp.size:
                cnt_df = cnt_df_tmp
            else:
                cnt_df = cnt_df.head(slice_size)
    else:
        cnt_df_tmp = cnt_df[
             cnt_df["pid"] > 0
        ]
        if cnt_df_tmp.size:
            cnt_df = cnt_df_tmp
        else:
            cnt_df = cnt_df.head(slice_size)
    cnt_df["pid"] = cnt_df.apply(
        lambda s: fuzz.ratio(s["s"], zh_first) * s["pid"]
    , axis = 1)
    cnt_df = cnt_df.sort_values(
        by = "pid", ascending = False
    )
    return cnt_df



#### may add a sim to maintain entity same category
#### add bm25 between en or (and) zh part with this perperty count
#@lru_cache(maxsize=52428800)
@timer()
def search_sim_entity_by_property_count(entityid, pid_relate_entity_df):
    assert hasattr(pid_relate_entity_df, "size")
    assert "pid" in pid_relate_entity_df.columns.tolist()
    assert "s" in pid_relate_entity_df.columns.tolist()

    forw_df = search_triples_with_parse(
    wiki_parser.document,
    find_query_direction_format(entityid, direction="forw"))

    entity_related_properties = forw_df["p"].map(
    lambda x: x if str(x).startswith("http://www.wikidata.org/prop") else np.nan
).dropna().map(
    lambda x: str(x).split("/")[-1]
).drop_duplicates().tolist()

    cnt_df = pid_relate_entity_df[
 pid_relate_entity_df["pid"].isin(
     entity_related_properties
 )
].explode("s").groupby("s")["pid"].apply(set).map(len).reset_index().sort_values(
    by = "pid", ascending = False
)

    return cnt_df

@timer()
def search_sim_entity_by_property_count_by_explode(entityid,
pid_relate_entity_explode_df):
    assert hasattr(pid_relate_entity_df, "size")
    assert "pid" in pid_relate_entity_df.columns.tolist()
    assert "s" in pid_relate_entity_df.columns.tolist()

    forw_df = search_triples_with_parse(
    wiki_parser.document,
    find_query_direction_format(entityid, direction="forw"))

    entity_related_properties = forw_df["p"].map(
    lambda x: x if str(x).startswith("http://www.wikidata.org/prop") else np.nan
).dropna().map(
    lambda x: str(x).split("/")[-1]
).drop_duplicates().tolist()

    cnt_df = pid_relate_entity_explode_df[
 pid_relate_entity_explode_df["pid"].isin(
     entity_related_properties
 )
].groupby("s")["pid"].apply(set).map(len).reset_index().sort_values(
    by = "pid", ascending = False
)

    return cnt_df

@timer()
def desc_matcher(x):
    return bool(re.findall(r"(P\d+)", x))

#### wd:Q, wdt:P need generalized.
@timer()
def sim_representation_decomp(en_sent, sparql_query, prefix_s, prefix_url_dict,
property_info_df,
pid_tuple_on_s_dict,
                              zh_linker_entities,
                              ner_fix,
                              zh_entity_db,
                              wiki_entity_db = wiki_entity_db,
                             entity_aug_size = 100, only_guess = False,
                             not_have_zh_rp_by_trans = True,
                             skip_no_db_zh_entity_str = False,
                             string_sent_length_ratio_threshold = 0.9,
                             ):
    assert string_sent_length_ratio_threshold is None or type(string_sent_length_ratio_threshold) == type(0.0)

    assert type(en_sent) == type("")
    assert type(sparql_query) == type("")
    kb_part_list = retrieve_all_kb_part(sparql_query, prefix_s, prefix_url_dict, fullfill_with_url=False)
    assert type(kb_part_list) == type([])

    kb_part_list = list(set(kb_part_list))
    entities = []
    properties = []
    others = []
    for ele in kb_part_list:
        if ele.startswith("wd:Q"):
            entities.append(ele)
        elif ele.startswith("wdt:P"):
            properties.append(ele)
        else:
            others.append(ele)

    for ele in others:
        if ":" in ele and ele.split(":")[-1].startswith("P"):
            if ele not in properties:
                properties.append(ele)

    entities, properties, others = map(lambda x: list(set(x)), [entities, properties, others])

    entity_prop_dict = dict(property_info_df[
    property_info_df["pid"].isin(
        list(map(lambda x: x.split(":")[-1] ,properties))
    )
    ][["pid", "en_info"]].values.tolist())

    req_ = {}
    for k, v in entity_prop_dict.items():
        assert type(v) == type([])
        if bool(v):
            req_[k] = v
    entity_prop_dict = req_

    for entityid in entities:
        #entity_prop_dict[entityid[3:]] = search_entity_rep_by_lang_filter(entityid[3:], lang = "en")
        en_ = search_entity_rep_by_lang_filter_in_db(entityid[3:], wiki_entity_db, lang = "en")
        if not en_:
            en_ = search_entity_rep_by_lang_filter_by_init_dict(entityid[3:], lang = "en")
        if en_:
            entity_prop_dict[entityid[3:]] = en_

    #### entity property desc string length filter
    if string_sent_length_ratio_threshold is None:
        string_sent_length_ratio_threshold = 10.0
    req_ = {}
    for k, v in entity_prop_dict.items():
        assert type(v) == type([])
        vv = list(filter(lambda x:
        (float(len(x)) / len(en_sent)) < string_sent_length_ratio_threshold
         , v))
        if bool(vv):
            req_[k] = vv
    entity_prop_dict = req_

    #guess_dict = guess_sim_representation(en_sent, entity_prop_dict)
    #guess_dict = guess_sim_representation_by_score(en_sent, entity_prop_dict)
    guess_dict = map_reduce_guess_sim_representation_by_score(en_sent, entity_prop_dict)
    if only_guess:
        return guess_dict

    zh_string_entity_id_dict = {}
    en_string_entity_id_dict = {}

    ##### guess aug by property and sim part
    entity_id_aug_dict = dict()
    for entityid in tqdm(entities):
        #### this should also take this entityid it self string into l,
        #### to generate keep above the bottom line
        #### i.e. l = l + zh

        l = search_sim_entity_by_property_count_by_dict_add_fuzz_f_by_db(entityid[3:],
                                                    pid_tuple_on_s_dict,
                                                    ner_fix,
                                                    zh_entity_db
                                                   ).head(entity_aug_size)["s"].tolist()
        l = list(filter(lambda x: not x.startswith("维基") and not x.startswith("維基"), l))

        #### this zh is the pivot string list to sort l (sim entity by property count)
        #zh = search_entity_rep_by_lang_filter(entityid[3:], "zh")
        zh = search_entity_rep_by_lang_filter_in_db(entityid[3:], wiki_entity_db, "zh")
        if not zh:
            zh = search_entity_rep_by_lang_filter_by_init_dict(entityid[3:], "zh")

        #zh = search_entity_rep_by_lang_filter_by_init_dict(entityid[3:], "zh")
        assert type(zh) == type([])
        if not zh and not_have_zh_rp_by_trans:
            #### new only use first entity id sim in emb as fix
            #en = search_entity_rep_by_lang_filter(entityid[3:], "en")
            en = search_entity_rep_by_lang_filter_in_db(entityid[3:], wiki_entity_db, "en")
            if not en:
                en = search_entity_rep_by_lang_filter_by_init_dict(entityid[3:], "en")

            #en = search_entity_rep_by_lang_filter_by_init_dict(entityid[3:], "en")
            assert type(en) == type([])
            if en:
                zh_ = []
                for en_ele in en:
                    assert type(en_ele) == type("")
                    zh_df = find_en_str_entityid_by_trans_near_linking(en_ele,
                                          zh_linker_entities, ner_fix, show_score_df=False
                                          )
                    assert hasattr(zh_df, "size")
                    if zh_df.size > 0:
                        zh_df.columns = ["token", "id_token_t2"]
                        zh_df["id"] = zh_df["id_token_t2"].map(lambda t2: t2[0])
                        zh_df["id_zh_entity_str"] = zh_df["id_token_t2"].map(lambda t2: t2[1])
                        zh_df = zh_df[["id", "id_zh_entity_str"]].explode("id_zh_entity_str").dropna().drop_duplicates()
                        first_id = zh_df["id"].iloc[0]
                        zh = zh_df[
                        zh_df["id"] == first_id
                        ]["id_zh_entity_str"].drop_duplicates().tolist()
                        assert type(zh) == type([])
                        zh_.extend(zh)
                zh = zh_

        for zh_ele in zh:
            zh_string_entity_id_dict[zh_ele] = entityid[3:]

        l = l + zh

        l = sorted(l, key = lambda x:
           max(map(lambda y: fuzz.ratio(x, y), zh))
          , reverse = True) if zh else l
        ll = []
        for e in l:
            if e not in ll:
                ll.append(e)
        l = ll
        entity_id_aug_dict[entityid[3:]] = l

    entity_id_aug_en_dict = dict()

    err_ = []
    for entityid in tqdm(entity_id_aug_dict.keys()):
        l = entity_id_aug_dict[entityid]
        assert type(l) == type([])
        req_en_l = []
        for zh_entity_str in l:
            entity_id_zh_string_t2 = None

            entity_id_zh_string_t2 = find_zh_str_entityid_by_db(zh_entity_str, wiki_entity_db)
            if entity_id_zh_string_t2 is None and not skip_no_db_zh_entity_str:
                try:
                    entity_id_zh_string_t2 = find_zh_str_entityid_by_linking(zh_entity_str, zh_linker_entities)
                except:
                    err_.append(zh_entity_str)
                    continue

            if type(entity_id_zh_string_t2) == type((1,)) and len(entity_id_zh_string_t2) == 2:
                entity_id = entity_id_zh_string_t2[0]
                assert type(entity_id) == type("")

                #entity_en_represent_list = search_entity_rep_by_lang_filter(entity_id, "en")
                #entity_en_represent_list = search_entity_rep_by_lang_filter_by_init_dict(entity_id, "en")
                entity_en_represent_list = \
                search_entity_rep_by_lang_filter_in_db(entity_id, wiki_entity_db, "en")
                if not entity_en_represent_list:
                    entity_en_represent_list = \
                    search_entity_rep_by_lang_filter_by_init_dict(entity_id, "en")


                for en_ele in entity_en_represent_list:
                    en_string_entity_id_dict[en_ele] = entity_id

                if type(entity_en_represent_list) == type([]) and entity_en_represent_list:
                    entity_en_representation = entity_en_represent_list[0]
                    if entity_en_representation not in req_en_l:
                        req_en_l.append(entity_en_representation)

        entity_id_aug_en_dict[entityid] = req_en_l


    return guess_dict, entity_id_aug_dict, entity_id_aug_en_dict, zh_string_entity_id_dict, en_string_entity_id_dict, err_

@timer()
def process_token(token, rm_char_list = ["?"]):
    for rm_c in rm_char_list:
        token = token.replace(rm_c, "")
    return token


@timer()
def phrase_validation(en_sent, phrase_string, rm_char_list = ["?"],
                        any_or_all = any, lower_it = True
                      ):
    ####print(en_sent, phrase_string)
    if lower_it:
        en_sent, phrase_string = map(lambda x: x.lower(), [en_sent, phrase_string])
    assert callable(any_or_all)
    assert type(en_sent) == type("")
    assert type(phrase_string) == type("")
    if phrase_string not in en_sent:
        return False

    assert phrase_string in en_sent
    def process_token(token):
        for rm_c in rm_char_list:
            token = token.replace(rm_c, "")
        return token

    en_sent = process_token(en_sent).strip()
    #phrase_string = process_token(phrase_string).strip()
    if not en_sent.strip():
        return False
    if phrase_string not in en_sent:
        return False

    sp_l = en_sent.split(phrase_string)
    assert len(sp_l) >= 2
    valid_list = []

    for i in range(1 ,len(sp_l)):
        left = sp_l[i - 1]
        right = sp_l[i]
        left_valid = (not left.strip()) or left.endswith(" ")
        right_valid = (not right.strip()) or right.startswith(" ")
        if left_valid and right_valid:
            valid_list.append(True)
        else:
            valid_list.append(False)
    return any_or_all(valid_list)

@timer()
def most_sim_token_in_sent(en_sent, entity_str, rm_char_list = ["?", '`']):
    def process_token(token):
        for rm_c in rm_char_list:
            token = token.replace(rm_c, "")
        return token

    sent_tokens = list(filter(lambda x: x.strip() ,
                              map(lambda y: process_token(y) ,en_sent.split(" "))
                             )
                      )

    if not sent_tokens:
        return None
    return sorted(sent_tokens, key = lambda x: fuzz.ratio(x, entity_str), reverse = True)[0]

@timer()
def recurrent_decomp_entity_str_by_en_sent(en_sent, entity_str, drop_dup_sent_tokens = True):
    entity_str_list = [entity_str]
    req = []
    while entity_str_list:
        e_str = entity_str_list.pop(0)

        most_sim_in_sent = most_sim_token_in_sent(en_sent, e_str)
        if type(most_sim_in_sent) != type(""):
            continue
        most_sim_in_e_str = most_sim_token_in_sent(e_str, most_sim_in_sent)
        if type(most_sim_in_e_str) != type(""):
            continue
        req.append(
            (most_sim_in_sent, most_sim_in_e_str)
        )

        res_x = set(filter(lambda xx: xx.strip() ,e_str.replace(most_sim_in_e_str, "").split(" ")))

        for ele in res_x:
            entity_str_list.append(ele)

    entity_sim_df = pd.DataFrame(req, columns = ["sent_token", "entity_token"])
    if drop_dup_sent_tokens:
        entity_sim_df = entity_sim_df[
        entity_sim_df.apply(lambda x:
        len(set(x.iloc[0].lower()).intersection(set(x.iloc[1].lower()))) > 0
        , axis = 1)
        ]
        if entity_sim_df.size > 0:
            entity_sim_df = drop_duplicates_by_col(entity_sim_df, "sent_token")
        else:
            return None
        #assert entity_sim_df.size > 0
        #assert entity_sim_df.size > 0

    return entity_sim_df

@timer()
def sp_string_by_desc_str_list(input_str, desc_str_list = ['of',
 'the','a','for','in','or','on','an','to','by','and','at','this','is','with','that','which','has','as',
 'from','de']):
    assert type(input_str) == type("")
    assert type(desc_str_list) == type([])
    need_sp_list = list(filter(lambda x: phrase_validation(input_str ,x.strip()), desc_str_list))
    if not need_sp_list:
        return [[input_str]]
    sent_tokens = list(filter(lambda x: x.strip() ,
                              map(lambda y: y ,input_str.split(" "))
                             )
                      )
    req = []
    for ele in sent_tokens:
        if ele in need_sp_list:
            req.append([])
        else:
            if not req:
                req.append([])
            req[-1].append(ele)
    return req

#### from produce_join_action in ner_trans only_fix_script_ser.py
@timer()
def produce_join_action(token_list, sp_char = [" ", "–"]):
    #token_list = list(filter(lambda y: y ,map(lambda x: x.strip().replace(" ", ""), token_list)))
    #token_list = list(filter(lambda y: y ,map(lambda x: x.strip(), token_list)))
    assert type(token_list) == type([])
    assert type(sp_char) == type([])
    if len(token_list) <= 1:
        #return [token_list]
        return token_list
    #action_nest_zip = product(*map(lambda _: ("+", "/") ,range(len(token_list) - 1)))
    action_nest_zip = product(*map(lambda _: sp_char ,range(len(token_list) - 1)))
    req = []
    for sep_l in action_nest_zip:
        #print(sep_l)
        text = ""
        for i, sep in enumerate(sep_l):
            text += token_list[i]
            #assert sep in ("+", "/")
            assert sep in sp_char

            text += sep
        assert i == (len(token_list) - 2)
        text += token_list[-1]
        req.append(text)
        #req.append(text.split("*****"))
    return req

@timer()
def eat_token_from_string(string, from_, num = 1):
    assert type(string) == type("")
    assert from_ in ["left", "right"]
    sp_char = [" ", "-"]
    c_l = re.split("[{}]".format("".join(sp_char)), string)
    c_l = list(filter(lambda x: x.strip(), c_l))
    assert type(c_l) == type([])
    if from_ == "left":
        return c_l[:num]
    else:
        return c_l[-1 * num:]

@timer()
def nearby_sent_by_token(sent, token, size = 3, lower_it = False):
    assert type(sent) == type("")
    assert type(token) == type("")
    if not phrase_validation(sent, token, lower_it = lower_it):
        return []
    l = sent.split(token)
    assert len(l) >= 2
    l = pd.Series(unzip_string(l, 2)).map(
        lambda t2: (eat_token_from_string(t2[0], "right", size),
                    eat_token_from_string(t2[1], "left", size))
    ).explode().tolist()
    return reduce(lambda a, b: a + b ,l) if l else []

@timer()
def fix_v_mapping(v_mapping, en_sent):
    assert type(v_mapping) == type({})
    assert len(v_mapping) == 1
    vk = list(v_mapping.keys())[0]
    if vk.strip() == "":
        return v_mapping
    if vk.strip().lower() not in en_sent.lower():
        return v_mapping
    vv = v_mapping[vk]
    vk = vk.strip()
    lower_eq_l = []
    for i in range(len(en_sent)):
        tail = en_sent[i:i+len(vk)]
        if len(tail) != len(vk):
            continue
        else:
            if vk.lower() == tail.lower():
                ###return {tail: vv}
                lower_eq_l.append(tail)

    assert lower_eq_l
    if len(lower_eq_l) == 1:
        return {lower_eq_l[0]: vv}
    phrase_validation_lower_eq_l = \
    list(filter(lambda x: phrase_validation(en_sent, x), lower_eq_l))
    if phrase_validation_lower_eq_l and len(phrase_validation_lower_eq_l) == 1:
        return {phrase_validation_lower_eq_l[0]: vv}

    #### for performance
    return {lower_eq_l[0]: vv}


@timer()
def guess_most_sim_pharse_in_en_sent(en_sent, entity_str, sent_token_list,
    fuzz_threshold = 70,
return_df = False, preprocess_sent = True):
    assert type(en_sent) == type("")
    assert type(entity_str) == type("")
    assert type(sent_token_list) == type([])
    def process_token(token, rm_char_list = ['"', "'", ","]):
        for rm_c in rm_char_list:
            token = token.replace(rm_c, "")
        return token

    if preprocess_sent:
        en_sent = process_token(en_sent)

    sent_token_list_ = []
    for ele in sent_token_list:
        if ele not in sent_token_list_:
            sent_token_list_.append(ele)
    sent_token_list = sent_token_list_

    for ele in sent_token_list:
        assert (ele in en_sent) or (process_token(ele) in en_sent)
    #### too long perm time if sent_token_list too long
    entity_str_list = entity_str.split(" ")
    if sent_token_list:
        sent_token_list_sorted = list(map(lambda x:
    sorted(sent_token_list, key = lambda xx: fuzz.ratio(xx, x), reverse = True)[0]
    , entity_str_list))
    else:
        sent_token_list_sorted = []


    if len(sent_token_list_sorted) == len(entity_str_list):
        sent_token_list_sorted_ = []
        for i in range(len(sent_token_list_sorted)):
            a = entity_str_list[i]
            b = sent_token_list_sorted[i]
            if b:
                b_c = b[0].upper() + b[1:]
                b_u = b.upper()
                b_l = b.lower()
                b_f = sorted([b, b_c, b_u, b_l], key = lambda x: fuzz.ratio(x, a), reverse = True)[0]
            else:
                b_f = b
            sent_token_list_sorted_.append(b_f)
        sent_token_list_sorted = sent_token_list_sorted_

    def list_eq(a, b, false_ratio_threshold = 0.2):
        assert type(a) == type([])
        assert type(b) == type([])
        if not a or not b:
            return False
        valid_list = list(map(lambda i: fuzz.ratio(a[i], b[i]) >= 90, range(len(a))))
        assert valid_list
        if not valid_list[0]:
            return False
        if not valid_list[-1]:
            return False
        if (float(sum(map(lambda x: not x, valid_list))) / len(valid_list)) > false_ratio_threshold:
            return False
        return True

    #print(sent_token_list_sorted)
    #print(entity_str_list)

    #if sent_token_list_sorted and sent_token_list_sorted == entity_str_list and " ".join(sent_token_list_sorted) == entity_str:
    if sent_token_list_sorted and list_eq(sent_token_list_sorted,
                                         entity_str_list
                                         ):
        sent_token_p_str_m = [" ".join(sent_token_list_sorted)]
    else:
        sent_token_p_str_m = sent_token_list + list(map(" ".join ,permutations(sent_token_list)))

    def process_c_token(c, en_sent, c_):
        assert type(c) == type("")
        assert type(en_sent) == type("")
        if c:

            sp_char = [" ", "-"]
            sent_l = re.split("[{}]".format("".join(sp_char)), en_sent)

            c_prod = list(map("".join ,product(*pd.Series(list(c)).map(
                                lambda x: (x.lower(), x.upper())
                                ).tolist())))
            assert bool(c_prod)
            in_num = sum(map(lambda x: x in
            sent_l
            , c_prod))
            assert in_num == 0 or in_num == 1 or in_num > 1
            if in_num == 1:
                c_f = list(filter(lambda x: x in sent_l, c_prod))[0]
            else:
                c_f_df = pd.DataFrame(pd.Series(c_prod).map(
                lambda x: (x, int(x in en_sent) * fuzz.ratio(x, en_sent))
                ).tolist())
                assert c_f_df.shape[1] == 2
                c_f_df.columns = ["token", "fuzz"]
                max_fuzz = c_f_df["fuzz"].max()
                if c_f_df[
                    c_f_df["fuzz"] == max_fuzz
                ].shape[0] != 1:
                    assert c_f_df[
                        c_f_df["fuzz"] == max_fuzz
                    ].shape[0] > 1
                    c_f_df["nearby_fuzz"] = c_f_df.apply(
                        lambda x:
                        fuzz.ratio(" ".join(nearby_sent_by_token(en_sent, x["token"])), c_)
                        ,axis = 1
                    )
                    c_f_df = c_f_df.sort_values(by = ["fuzz", "nearby_fuzz"], ascending = False)
                else:
                    c_f_df = c_f_df.sort_values(by = ["fuzz"], ascending = False)
                c_f = c_f_df["token"].iloc[0]

        else:
            c_f = c
        c = c_f
        return c

    req = []
    for ele in sent_token_p_str_m:
        #a = get_match_intersection(ele, en_sent)
        a = get_match_intersection_fill_gap(ele, en_sent, en_sent)
        if a is None:
            a = ""
        #b = get_match_intersection(en_sent ,ele)
        b = get_match_intersection_fill_gap(en_sent ,ele, en_sent)
        if b is None:
            b = ""
        c = a if len(a) >= len(b) else b

        if c not in en_sent:
            fix_c_mapping_dict = fix_v_mapping({c:c}, en_sent)
            assert type(fix_c_mapping_dict) == type({})
            assert len(fix_c_mapping_dict) == 1
            fix_c_mapping_k = list(fix_c_mapping_dict.keys())[0]
            if fix_c_mapping_k in en_sent:
                c = fix_c_mapping_k
            else:
                pass

        req.append(
            (ele, c.strip())
        )
    df = pd.DataFrame(req, columns = ["sent_token_p_str", "inter_str"])

    df["is_phrase"] = df["inter_str"].map(
        lambda x: phrase_validation(en_sent, x.strip()) if (en_sent.strip() and x.strip()) else False
    )


    df["inter_str_len"] = df["inter_str"].map(len)
    df["fuzz"] = df.apply(lambda s:
    fuzz.ratio(s["sent_token_p_str"], entity_str) +\
    fuzz.ratio(s["sent_token_p_str"], s["inter_str"])
    , axis = 1)
    df["score"] = df["inter_str_len"] * df["fuzz"]
    df = df.sort_values(by = "fuzz", ascending = False)
    #print(df)

    df = df[
        df.apply(
            lambda s: s["is_phrase"] and s["fuzz"] >= fuzz_threshold * 2
        , axis = 1)
    ]

    if return_df:
        return df

    if not df.size:
        return {"": entity_str}

    return {df["inter_str"].iloc[0]: entity_str}

@timer()
def map_guess_dict_to_in_sent_mapping(en_sent ,guess_dict):
    assert type(en_sent) == type("")
    assert type(guess_dict) == type({})
    for k, v in guess_dict.items():
        #assert len(set(v)) == 1
        assert len(set(v)) <= 2
        assert len(set(v)) >= 1
        pass

    req = dict(map(lambda t2: (
        t2[0], sorted(t2[1], key = len, reverse = True)[0]
    ), guess_dict.items()))

    req_ = {}
    for k, v in req.items():
        assert type(v) == type("")
        if v in en_sent:
            v_mapping = {v: v}
            #print(0, v_mapping)
            req_[k] = v_mapping
            continue
        else:
            v_mapping_fixed = fix_v_mapping({v: v}, en_sent)
            vkf = list(v_mapping_fixed.keys())[0]
            if vkf in en_sent:
                v_mapping = v_mapping_fixed
                #print(1, v_mapping)
                req_[k] = v_mapping
                continue

            sim_df = recurrent_decomp_entity_str_by_en_sent(en_sent, v)
            if sim_df is None:
                ##### sclice_empty
                v_mapping = {"": v}
                #print(2, v_mapping)
                req_[k] = v_mapping
                continue
            else:
                assert hasattr(sim_df, "size")

                v_mapping_0 = guess_most_sim_pharse_in_en_sent(en_sent,
                                         v
                                         , sim_df["sent_token"].tolist(),
                                         preprocess_sent = True
                                        )
                v_mapping_1 = guess_most_sim_pharse_in_en_sent(en_sent,
                                         v
                                         , sim_df["sent_token"].tolist(),
                                         preprocess_sent = False
                                        )

                if v_mapping_0 == v_mapping_1:
                    v_mapping = v_mapping_0
                else:

                    v_mapping_0_valid = True
                    v_mapping_1_valid = True
                    for kk in v_mapping_0.keys():
                        if kk not in en_sent:
                            v_mapping_0_valid = False
                            break
                    for kk in v_mapping_1.keys():
                        if kk not in en_sent:
                            v_mapping_1_valid = False
                            break

                    assert v_mapping_0_valid or v_mapping_1_valid
                    if v_mapping_0_valid and not v_mapping_1_valid:
                        v_mapping = v_mapping_0
                    elif v_mapping_1_valid and not v_mapping_0_valid:
                        v_mapping = v_mapping_1
                    else:
                        assert v_mapping_0_valid and v_mapping_1_valid
                        if sum(map(len, v_mapping_0.keys())) > sum(map(len, v_mapping_1.keys())):
                            v_mapping = v_mapping_0
                        elif sum(map(len, v_mapping_1.keys())) > sum(map(len, v_mapping_0.keys())):
                            v_mapping = v_mapping_1
                        else:
                            assert sum(map(len, v_mapping_1.keys())) == sum(map(len, v_mapping_0.keys()))
                            if sum(map(len, v_mapping_0.values())) > sum(map(len, v_mapping_1.values())):
                                v_mapping = v_mapping_0
                            elif sum(map(len, v_mapping_1.values())) > sum(map(len, v_mapping_0.values())):
                                v_mapping = v_mapping_1
                            else:
                                assert sum(map(len, v_mapping_1.values())) == sum(map(len, v_mapping_0.values()))
                                if sum(map(lambda vvv: fuzz.ratio(vvv, en_sent), v_mapping_0.values())) > \
                                sum(map(lambda vvv: fuzz.ratio(vvv, en_sent), v_mapping_1.values())):
                                    v_mapping = v_mapping_0
                                elif sum(map(lambda vvv: fuzz.ratio(vvv, en_sent), v_mapping_1.values())) > \
                                sum(map(lambda vvv: fuzz.ratio(vvv, en_sent), v_mapping_0.values())):
                                    v_mapping = v_mapping_1
                                else:
                                    v_mapping = v_mapping_0
                #print(3, v_mapping)
                if list(v_mapping.keys())[0] in en_sent:
                    req_[k] = v_mapping
                    continue

                #print("before :")
                #print(v_mapping)
                v_mapping_fixed = fix_v_mapping(v_mapping, en_sent)
                #print("after :")
                #print(afetr)
                vkf = list(v_mapping_fixed.keys())[0]
                if vkf in en_sent:
                    v_mapping = v_mapping_fixed
                #print(4, v_mapping)
        assert type(v_mapping) == type({})
        for kk in v_mapping.keys():
            assert kk in en_sent
        req_[k] = v_mapping
    #return req_

    #### drop overlap
    req = []
    for k, v in req_.items():
        id_ = k
        inter_str = list(v.keys())[0]
        entity_str = list(v.values())[0]
        req.append((id_, inter_str, entity_str))

    req_df = pd.DataFrame(req, columns = ["id", "inter_str", "entity_str"])
    req_df["fuzz"] = req_df["inter_str"].map(
        lambda x: fuzz.ratio(x, en_sent)
    )
    req_df["is_entity"] = req_df["id"].map(
        lambda x: x.startswith("Q")
    )
    req_df = req_df.sort_values(by = ["is_entity", "fuzz"], ascending = False)
    req = []
    valid_str = en_sent
    for i, r in req_df.iterrows():
        if r["inter_str"] and r["inter_str"] in valid_str:
            req.append(r)
            valid_str = valid_str.replace(r["inter_str"], "*")
        else:
            r["inter_str"] = ""
            r["fuzz"] = fuzz.ratio(r["inter_str"], en_sent)
            req.append(r)
    req_df = pd.DataFrame(req)
    req_df = req_df.sort_values(by = ["is_entity", "fuzz"], ascending = False)
    return req_df

@timer()
def unzip_one_t2(t2, desc_str_list = ['of',
 'the','a','for','in','or','on','an','to','by','and','at','this','is','with','that','which','has','as',
 'from','de']):
    assert type(t2) == type((1,))
    assert len(t2) == 2
    k, v = t2
    assert type(k) == type("")
    assert type(v) == type([])
    l2 = list(map(
    lambda x: sp_string_by_desc_str_list(x, desc_str_list = desc_str_list)
    , v))
    assert len(l2) == 2
    p = product(*l2)
    v_l = list(map(lambda l: (" ".join(l[0]), " ".join(l[1])), p))
    return t2[0] ,v_l

@timer()
def slice_guess_dict_in_sent_mapping_df(guess_dict_in_sent_mapping_df, fuzz_threshold = 20.0):
    assert hasattr(guess_dict_in_sent_mapping_df, "size")
    if not guess_dict_in_sent_mapping_df.size:
        return guess_dict_in_sent_mapping_df
    cols = guess_dict_in_sent_mapping_df.columns.tolist()
    assert "inter_str" in cols
    assert "entity_str" in cols
    assert "fuzz" in cols
    req = []
    all_id_set = set(guess_dict_in_sent_mapping_df["id"].values.tolist())
    for id_, df in guess_dict_in_sent_mapping_df.groupby("id"):
        df_empty = df[
            df["fuzz"] == 0
        ]
        df_nan_empty = df[
            df["fuzz"] >= fuzz_threshold
        ]
        if df_nan_empty.size:
            df = pd.concat([df_empty, df_nan_empty], axis = 0)
        else:
            if df_empty.size:
                df = df_empty
            pass
        req.append(df)
    return pd.concat(req, axis = 0)

@timer()
def map_guess_dict_to_in_sent_mapping_multi_times(en_sent, guess_dict,
    desc_str_list = ['of',
     'the','a','for','in','or','on','an','to','by','and','at','this','is','with','that','which','has','as',
     'from','de', '-']
):
    assert type(guess_dict) == type({})
    guess_dict = dict(filter(
        lambda t2: sum(map(lambda x: len(x.strip()) ,t2[1])) > 0
    , guess_dict.items()))

    guess_l_dict = dict(map(
        lambda t2: unzip_one_t2(t2, desc_str_list = desc_str_list)
        , guess_dict.items()))
    #return guess_l_dict
    all_keys = list(guess_l_dict.keys())
    all_vals = list(map(lambda k: guess_l_dict[k], all_keys))
    p_l = list(product(*all_vals))
    if guess_dict not in p_l:
        p_l.append(
            tuple(map(lambda k: guess_dict[k], all_keys))
        )
    p_l_dict = []
    p_l_guess_dict_in_sent_mapping_df = []
    guess_dict_in_sent_mapping_df = None
    if p_l:
        assert all(map(lambda x: len(x) == len(all_keys), p_l))
        p_l_df = pd.DataFrame(p_l)
        p_l_df.columns = all_keys
        p_l_dict = p_l_df.apply(lambda x: x.to_dict(), axis = 1).values.tolist()

        p_l_dict = list(filter(lambda d:
        all(map(lambda t2: sum(map(lambda x: len(x.strip()) ,t2[1])) > 0, d.items()))
        , p_l_dict))
        assert bool(p_l_dict)

        p_l_guess_dict_in_sent_mapping_df = list(map(
            lambda gd: map_guess_dict_to_in_sent_mapping(en_sent, gd)
            , p_l_dict))
        guess_dict_in_sent_mapping_df = pd.concat(p_l_guess_dict_in_sent_mapping_df,
                  axis = 0).drop_duplicates().sort_values(by = ["is_entity", "fuzz"], ascending = False)
        guess_dict_in_sent_mapping_df = slice_guess_dict_in_sent_mapping_df(guess_dict_in_sent_mapping_df)
        guess_dict_in_sent_mapping_df = guess_dict_in_sent_mapping_df.drop_duplicates().sort_values(by = ["is_entity", "fuzz"], ascending = False)
        guess_dict_in_sent_mapping_df = drop_duplicates_by_col(guess_dict_in_sent_mapping_df, on_col = "id")
    return guess_dict_in_sent_mapping_df

@timer()
def generate_all_in_db_id_filling(sparql_query, prefix_s, prefix_url_dict):
    kb_part_list = retrieve_all_kb_part(sparql_query, prefix_s, prefix_url_dict, fullfill_with_url=False)
    if not kb_part_list:
        return {}
    prefix_id_df = pd.DataFrame(pd.Series(kb_part_list).map(
        lambda x: x.split(":")
    ).values.tolist(), columns = ["prefix", "id"])
    return prefix_id_df.groupby("id")["prefix"].apply(set).map(list).to_dict()

#### reverse data prepare with sim_representation_decomp
@timer()
def produce_sim_representation_reconstruct_df(
    en_sent, sparql_query,
    property_info_df,

    guess_dict,
    entity_id_aug_dict, entity_id_aug_en_dict,
    zh_string_entity_id_dict, en_string_entity_id_dict
):
    assert type(en_sent) == type("")
    assert type(sparql_query) == type("")
    assert hasattr(property_info_df, "size")
    assert type(guess_dict) == type({})
    assert type(entity_id_aug_dict) == type({})
    assert type(entity_id_aug_en_dict) == type({})
    assert type(zh_string_entity_id_dict) == type({})
    assert type(en_string_entity_id_dict) == type({})

    entity_id_aug_en_dict_f = dict(map(lambda t2: (t2[0],
                                           list(filter(lambda x: not x.lower().startswith("wiki") ,t2[1]))
                                           ),
                                entity_id_aug_en_dict.items()))

    entity_id_aug_en_dict_f = dict(map(lambda t2: (t2[0],
                   list(map(lambda en: (en_string_entity_id_dict[en], en) ,t2[1]))
                   ),
        entity_id_aug_en_dict_f.items()))

    #### this add property step can also add some rule sim property
    for idx, r in property_info_df.iterrows():
        if r["pid"] in guess_dict.keys():
            entity_id_aug_en_dict_f[
            r["pid"]
        ] = list(map(lambda x: (r["pid"], x), r["en_info"]))

    #### entity and preproty string mapping
    #guess_dict_in_sent_mapping_df = map_guess_dict_to_in_sent_mapping(en_sent, guess_dict)
    guess_dict_in_sent_mapping_df = map_guess_dict_to_in_sent_mapping_multi_times(en_sent,
    guess_dict)

    id_prefix_dict = generate_all_in_db_id_filling(sparql_query, prefix_s, prefix_url_dict)
    assert type(id_prefix_dict) == type({})
    type_prefix_dict = pd.DataFrame(list(map(lambda t2: (t2[0][:1], t2[1]),id_prefix_dict.items()))).explode(1).groupby(0)[1].apply(set).map(list).to_dict()
    assert type(type_prefix_dict) == type({})

    guess_dict_in_sent_mapping_df["id_in_db"] = guess_dict_in_sent_mapping_df["id"].map(
            lambda id_: list(map(lambda prefix: "{}:{}".format(prefix, id_) ,id_prefix_dict[id_]))
    )
    guess_dict_in_sent_mapping_df = guess_dict_in_sent_mapping_df.explode("id_in_db").dropna().drop_duplicates()


    guess_dict_in_sent_mapping_df["aug_l"] = guess_dict_in_sent_mapping_df["id"].map(
        lambda id_: entity_id_aug_en_dict_f.get(id_, [])
    )

    #return guess_dict_in_sent_mapping_df

    guess_dict_in_sent_mapping_df_exploded = guess_dict_in_sent_mapping_df.explode("aug_l").sort_values(
    by = "fuzz", ascending = False
)
    guess_dict_in_sent_mapping_df_exploded["aug_id"] = guess_dict_in_sent_mapping_df_exploded["aug_l"].map(
        lambda t2: t2[0] if type(t2) == type((1,)) else t2
    )

    guess_dict_in_sent_mapping_df_exploded = guess_dict_in_sent_mapping_df_exploded.dropna()

    guess_dict_in_sent_mapping_df_exploded["aug_id_in_db"] = guess_dict_in_sent_mapping_df_exploded["aug_id"].map(
            lambda id_: list(map(lambda prefix: "{}:{}".format(prefix, id_) ,type_prefix_dict[id_[:1]]))
    )
    guess_dict_in_sent_mapping_df_exploded = guess_dict_in_sent_mapping_df_exploded.explode("aug_id_in_db").dropna().drop_duplicates()
    ##### aug maintain same prefix
    guess_dict_in_sent_mapping_df_exploded = guess_dict_in_sent_mapping_df_exploded[
    guess_dict_in_sent_mapping_df_exploded.apply(
        lambda s: s["id_in_db"].split(":")[0] == s["aug_id_in_db"].split(":")[0], axis = 1
    )]
    guess_dict_in_sent_mapping_df_exploded = guess_dict_in_sent_mapping_df_exploded[
    guess_dict_in_sent_mapping_df_exploded.apply(
        lambda s: s["aug_id_in_db"].split(":")[1].endswith(s["aug_id"]), axis = 1
    )]
    guess_dict_in_sent_mapping_df_exploded = guess_dict_in_sent_mapping_df_exploded.drop_duplicates().dropna()

    guess_dict_in_sent_mapping_df_exploded["aug_str"] = guess_dict_in_sent_mapping_df_exploded["aug_l"].map(
        lambda t2: t2[1] if type(t2) == type((1,)) else t2
    )

    guess_dict_in_sent_mapping_df_exploded["aug_fuzz"] = guess_dict_in_sent_mapping_df_exploded.apply(
        lambda s: fuzz.ratio(s["aug_str"], s["inter_str"]) if type(s["aug_str"]) == type("") else 0.0,
        axis = 1
    )

    guess_dict_in_sent_mapping_df_exploded = guess_dict_in_sent_mapping_df_exploded.sort_values(
        by = ["fuzz", "aug_fuzz"], ascending = False
    )

    return guess_dict_in_sent_mapping_df_exploded

@timer()
def maintain_entity_cut_on_en_sent(en_sent, maintain_list):
    assert type(en_sent) == type("")
    assert type(maintain_list) == type([])
    req = []
    pivot_string = en_sent
    while pivot_string:
        if maintain_list:
            head = sorted(filter(
                lambda x: pivot_string.startswith(x)
            , maintain_list), key = len, reverse = True)
            if head:
                head = head[0].strip()
                pivot_string = pivot_string[len(head):].strip()
            else:
                l = pivot_string.split(" ")
                head = l[0].strip()
                pivot_string = " ".join(l[1:]).strip()
        else:
            l = pivot_string.split(" ")
            head = l[0].strip()
            pivot_string = " ".join(l[1:]).strip()
        req.append(head)
    return req

@timer()
def one_row_aug(en_sent, sparql_query ,row, rp_entity_string_by_id = False):
    assert hasattr(row, "tolist")
    aug_en_sent = en_sent
    aug_sparql_query = sparql_query
    row_l = row.tolist()
    for t4 in row_l:
        assert len(t4) == 4
        ori_id, aug_id, ori_str, aug_str = t4

        if ori_str:
            #aug_en_sent = aug_en_sent.replace(ori_str, aug_str)
            aug_en_sent_tokens = maintain_entity_cut_on_en_sent(
                aug_en_sent, [ori_str]
            )
            if not rp_entity_string_by_id:
                aug_en_sent = " ".join(map(lambda x: aug_str if x == ori_str else x, aug_en_sent_tokens))
            else:
                aug_en_sent = " ".join(map(lambda x: aug_id if x == ori_str else x, aug_en_sent_tokens))
        aug_sparql_query = aug_sparql_query.replace(ori_id, aug_id)
        #print("tmp ")
        #print(aug_sparql_query)

    return aug_en_sent, aug_sparql_query

@timer()
def drop_duplicates_by_col(df, on_col = "aug_sparql_query"):
    assert hasattr(df, "size")
    assert on_col in df.columns.tolist()
    req = []
    set_ = set([])
    for i, r in df.iterrows():
        if r[on_col] not in set_:
            set_.add(r[on_col])
            req.append(r)
    return pd.DataFrame(req)

#### some filter may on property
@timer()
def sim_representation_reconstruct_by_df(en_sent, sparql_query, guess_dict_in_sent_mapping_df_exploded,
                                        aug_times = 100000,
                                        retrieve_diff_query = True,
                                        rp_entity_string_by_id = False,
                                        ):
    assert type(en_sent) == type("")
    assert type(sparql_query) == type("")
    assert hasattr(guess_dict_in_sent_mapping_df_exploded, "size")

    req = {}
    for gp_id_in_db, gp_df in guess_dict_in_sent_mapping_df_exploded.groupby("id_in_db"):
        aug_id_str_list = gp_df.apply(
            lambda s: (s["aug_id_in_db"], s["inter_str"], s["aug_str"])
            , axis = 1).values.tolist()
        req[gp_id_in_db] = aug_id_str_list

    d = req
    k_l = list(d.keys())
    v_l = list(map(lambda k: d[k], k_l))

    p = product(*v_l)
    req = []
    set_ = set([])
    for i, t4 in enumerate(p):
        if len(set_) > aug_times:
            break

        aug_c = list(map(lambda t2: t2[0] + t2[1] ,zip(map(lambda x: (x,) ,k_l), t4)))
        if aug_c not in req:
            req.append(aug_c)

        set_.add(
            tuple(filter(lambda t4: t4[2].strip() ,aug_c))
        )


    prod_df = pd.DataFrame(req)
    #return prod_df

    prod_s = prod_df.apply(
        lambda r: one_row_aug(en_sent, sparql_query ,r, rp_entity_string_by_id = rp_entity_string_by_id)
        , axis = 1)
    prod_df = pd.DataFrame(prod_s)
    prod_df = prod_df.drop_duplicates()
    prod_df.columns = ["aug"]
    prod_df["aug_en_sent"] = prod_df["aug"].map(lambda x: x[0])
    prod_df["aug_en_sent"] = prod_df["aug_en_sent"].map(
        lambda x: x.replace("  ", " ").replace("  ", " ").replace("  ", " ")
    )
    prod_df["aug_en_sent"] = prod_df["aug_en_sent"].map(
        repeat_to_one_on_en
    )
    prod_df = drop_duplicates_by_col(prod_df, "aug_en_sent")

    prod_df["aug_sparql_query"] = prod_df["aug"].map(lambda x: x[1])
    prod_df["fuzz"] = prod_df["aug_en_sent"].map(
        lambda x: fuzz.ratio(x, en_sent)
    )
    prod_df = prod_df.sort_values(by = "fuzz", ascending = False)[["aug_en_sent", "aug_sparql_query", "fuzz"]]
    prod_df = prod_df.drop_duplicates()
    if retrieve_diff_query:
        prod_df = drop_duplicates_by_col(prod_df)
    return prod_df

#### some property not in ori en_sent, in literal meaning
#### generate a insert on it.
@timer()
def seek_need_insert_property_list_from_df(guess_dict_in_sent_mapping_df_exploded):
    property_df = guess_dict_in_sent_mapping_df_exploded[
        ~guess_dict_in_sent_mapping_df_exploded["is_entity"]
    ]
    if not property_df.size:
        return []
    p_list = property_df[
        property_df["inter_str"].map(
            lambda x: not x.strip()
        )
    ]["id"].drop_duplicates().tolist()
    if not p_list:
        return []
    assert all(map(lambda x: x.startswith("P"), p_list))
    return p_list

def repeat_to_one_on_en(en_sent):
    req = []
    for ele in en_sent.split(" "):
        if not req:
            req.append(ele)
        else:
            if ele == req[-1]:
                continue
            else:
                req.append(ele)
    return " ".join(filter(lambda x: x.strip(), req))

#### maintain_phrase_list
#### other than add entity and preperty string
#### may add some common representation
#### such as "How many" and so on
#### if set desc_str_list to empty, skip desc validation

#### do some stats on maintain_phrase_list of
#### common phrase and desc_str_list
#### (same function as sp_tokens in eval_single_entity
#### in ner_trans)

#### return val may be a list or df
#### if list should use only one.
@timer()
def produce_all_insert_s(en_sent, en_info,
                         ner_fix,
                         maintain_phrase_list = [],
                        rm_char_list = ["?"],
                        desc_str_list = ['of',
 'the','a','for','in','or','on','an','to','by','and','at','this','is','with','that','which','has','as',
 'from','de']
                        ):
    #### maintain_phrase_list = ["Captain America", "How many"]
    #### en_sent: 'How many dimensions have a Captain America?'
    #### en_info: ['universe',
    #### 'featured in universe',
    #### 'appears in universe',
    #### 'cycle', 'in cycle']
    assert type(en_sent) == type("")
    assert type(en_info) == type([])
    assert type(maintain_phrase_list) == type([])
    if maintain_phrase_list:
        for ele in maintain_phrase_list:
            assert ele in en_sent

    en_info_used = list(filter(
        lambda x: x not in en_sent
        , en_info))

    if desc_str_list:
        en_info_used = list(filter(
        lambda x: any(map(lambda y: y in x.split(" "), desc_str_list))
        , en_info_used))

    if not en_info_used:
        return [en_sent]

    def process_token(token):
        for rm_c in rm_char_list:
            token = token.replace(rm_c, "")
        return token

    en_sent = process_token(en_sent).strip()

    en_tokens = find_max_len_cut_b_with_entity_maintain_j(
'', en_sent,
    maintain_phrase_list
)[1]

    #en_tokens = list(filter(lambda x: x.strip(), en_sent.split(" ")))

    if not en_tokens:
        return [en_sent]

    en_info_used = sorted(en_info_used, key = lambda x: len(x.split(" ")) * fuzz.ratio(x, en_sent),
                         reverse = True
                         )

    #return en_info_used, en_tokens

    head_it_list = list(map(
        lambda x: " ".join([x] + en_tokens)
        , en_info_used))

    tail_it_list = list(map(
        lambda x: " ".join(en_tokens + [x])
        , en_info_used))

    #### if desc_str_list not empty, only use head tail insert.
    if len(en_tokens) <= 1 or desc_str_list:
        mid_it_list = []
    else:
        mid_it_list = []
        for en_info_token in en_info_used:
            for i in range(1 ,len(en_tokens)):
                en_tokens_cp = deepcopy(en_tokens)
                en_tokens_cp.insert(i, en_info_token)
                mid_it_list.append(en_tokens_cp)
    mid_it_list = list(map(" ".join, mid_it_list))

    thm_it_s = pd.Series(tail_it_list + head_it_list + mid_it_list).map(lambda x: x.strip()).drop_duplicates()
    thm_it_df = pd.DataFrame(thm_it_s)
    thm_it_df = thm_it_df.drop_duplicates()
    thm_it_df.columns = ["en_sent_inserted"]
    thm_it_df["en_sent_inserted"] = thm_it_df["en_sent_inserted"].map(
        lambda x: x.replace("  ", " ").replace("  ", " ").replace("  ", " ")
    )
    thm_it_df["en_sent_inserted"] = thm_it_df["en_sent_inserted"].map(
        repeat_to_one_on_en
    )
    thm_it_df = drop_duplicates_by_col(thm_it_df, "en_sent_inserted")
    thm_it_df["fuzz"] = thm_it_df["en_sent_inserted"].map(
        lambda x: fuzz.ratio(en_sent, x)
    )
    thm_it_df = thm_it_df.sort_values(by = "fuzz", ascending = False)

    score_arr = perm_top_sort(en_sent, thm_it_df["en_sent_inserted"].tolist(),
                          ner_fix.sim_model, return_score = True)

    if type(score_arr) == type(""):
        #### only one not compare
        #### maintain zh_l
        assert len(thm_it_df) == 1
        score_arr = np.asarray([1.0 ,1.0])

    thm_it_df["sim_score"] = score_arr[1:]
    thm_it_df["score"] = thm_it_df.apply(lambda s: s["fuzz"] * s["sim_score"], axis = 1)
    thm_it_df = thm_it_df.sort_values(by = "score", ascending = False)
    thm_it_df = thm_it_df.drop_duplicates()
    thm_it_df = drop_duplicates_by_col(thm_it_df, "en_sent_inserted")
    return thm_it_df

@timer()
def produce_all_insert_s_two_times(
en_sent, en_info,
                         ner_fix,
                         maintain_phrase_list = [],
                        rm_char_list = ["?"],
                        desc_str_list = ['of',
 'the','a','for','in','or','on','an','to','by','and','at','this','is','with','that','which','has','as',
 'from','de']
):
    df0 = produce_all_insert_s(
        en_sent, en_info,
        ner_fix,
        maintain_phrase_list = maintain_phrase_list,
        desc_str_list = desc_str_list
    )
    assert type(df0) == type([]) or hasattr(df0, "size")

    df1 = produce_all_insert_s(
        en_sent, en_info,
        ner_fix,
        maintain_phrase_list = maintain_phrase_list,
        desc_str_list = []
    )
    assert type(df1) == type([]) or hasattr(df1, "size")

    if hasattr(df0, "size") and hasattr(df1, "size"):
        df = pd.concat([df0, df1], axis = 0)
        df = df.sort_values(by = "score", ascending = False)
        df = drop_duplicates_by_col(df, "en_sent_inserted")
        return df
    else:
        df_list = list(filter(lambda x: hasattr(x, "size"), [df0, df1]))
        if df_list:
            return df
        return list(set(df0 + df1))

@timer()
def perform_all_insert_on_need_insert_p_list(en_sent,
                                             ner_fix,
                                            property_info_df,
                                             guess_dict_in_sent_mapping_df_exploded,
                                             aug_df, need_insert_p_list,
                                             aug_size = 100
                                            ):
    assert type(en_sent) == type("")
    assert hasattr(aug_df, "size")
    assert type(need_insert_p_list) == type([])
    if not need_insert_p_list or aug_df.size == 0:
        return None

    aug_df = aug_df.copy().iloc[:aug_size, :]

    maintain_phrase_list = produce_maintain_list(en_sent, guess_dict_in_sent_mapping_df_exploded)

    now_sent = en_sent.strip()
    for pid in need_insert_p_list:
        en_info = dict(property_info_df[["pid", "en_info"]].values.tolist()).get(pid, [])
        if not en_info:
            continue
        insert_2_df = produce_all_insert_s_two_times(
                en_sent,
                en_info,
                ner_fix,
        maintain_phrase_list = maintain_phrase_list,
        )

        if hasattr(insert_2_df, "size"):
            sent_trans = insert_2_df["en_sent_inserted"].iloc[0]
        else:
            assert type(insert_2_df) == type([]) and bool(insert_2_df)
            sent_trans = insert_2_df[0]
        sent_trans = sent_trans.strip()
        if now_sent == sent_trans:
            continue

        guess_dict = map_reduce_guess_sim_representation_by_score(
                sent_trans,
                    {pid: en_info}
        )
        in_en_sent_df = map_guess_dict_to_in_sent_mapping(
            sent_trans
            ,guess_dict
        )
        if not hasattr(in_en_sent_df, "size"):
            continue
        if in_en_sent_df.size == 0:
            continue
        inter_str = in_en_sent_df["inter_str"].iloc[0]

        req = []
        for ele in tqdm(aug_df["aug_en_sent"].tolist()):
            insert_2_aug_df = produce_all_insert_s_two_times(
            ele,
            [inter_str],
            ner_fix,
            [],
        )
            if hasattr(insert_2_aug_df, "size"):
                sent_aug_trans = insert_2_aug_df["en_sent_inserted"].iloc[0]
            else:
                assert type(insert_2_aug_df) == type([]) and bool(insert_2_aug_df)
                sent_aug_trans = insert_2_aug_df[0]
            assert type(sent_aug_trans) == type("")
            req.append(sent_aug_trans)
        assert len(req) == aug_df.shape[0]
        aug_df["aug_en_sent"] = req
    return aug_df

@timer()
def unzip_string(x, size = 2):
    if len(x) <= size:
        return [x]
    req = []
    for i in range(len(x) - size + 1):
        req.append(x[i: i + size])
    return req

@timer()
def generate_head_df(df, question_col = "question",
    head_size = 30,  wn = WordNetLemmatizer()
):
    assert question_col in df.columns.tolist()

    head_df = pd.DataFrame(df[question_col].map(
    lambda x: x.split(" ")[:5]
).map(unzip_string).explode().map(tuple).map(
    lambda x: map(lambda y:
                  lemmatize_one_token(y.lower())
                  , x)
).map(tuple).value_counts().head(head_size).index.tolist())

    common_heads_0 = head_df.iloc[:, 0].tolist()
    common_heads_1 = head_df.iloc[:, 1].tolist()
    _0 = []
    _1 = []
    for ele in common_heads_0:
        if ele not in _0:
            _0.append(ele)
    for ele in common_heads_1:
        if ele not in _1:
            _1.append(ele)
    common_heads_0 = _0
    common_heads_1 = _1
    return common_heads_0, common_heads_1

#### set common_heads_0, common_heads_1 from generate_head_df
df = load_data()
common_heads_0, common_heads_1 = generate_head_df(df)
@timer()
def produce_maintain_list(en_sent, guess_dict_in_sent_mapping_df_exploded, rm_char_list = ["?"],
                        common_heads_0 = common_heads_0,
                          common_heads_1 = common_heads_1,
                          wn = WordNetLemmatizer()
                         ):
    def process_token(token):
        for rm_c in rm_char_list:
            token = token.replace(rm_c, "")
        return token

    inter_str_list = guess_dict_in_sent_mapping_df_exploded["inter_str"].drop_duplicates().map(
    lambda x: x.strip() if x.strip() else np.nan
).dropna().drop_duplicates().tolist()

    sent_tokens = list(filter(lambda x: x.strip() ,
                              map(lambda y: process_token(y) ,en_sent.split(" "))
                             )
                      )

    if sent_tokens:
        common_heads_list = list(filter(lambda x:
                                        lemmatize_one_token(x.lower())
                                        in common_heads_0 + common_heads_1
                            , sent_tokens))

        head_list = list(filter(lambda y: y in en_sent ,map(lambda x: " ".join(x) ,
                        unzip_string(common_heads_list, 2))))
        if not head_list:
            common_heads_list = list(filter(lambda x:
                                            lemmatize_one_token(x.lower())
                                            in common_heads_0
                            , sent_tokens))
            head_list = list(filter(lambda y: y in en_sent ,map(lambda x: " ".join(x) ,
                        unzip_string(common_heads_list, 1))))
    else:
        head_list = []


    req = head_list + inter_str_list
    need_list = []
    for ele in req:
        if ele not in need_list:
            need_list.append(ele)
    if need_list:
        for ele in need_list:
            assert ele in en_sent

    return need_list

@timer()
def generate_all_entity_id_needed_in_df(df, query_col = "sparql_wikidata",
    head_pattern = "wd:Q"
):
    assert query_col in df.columns.tolist()
    all_entity_s = df[query_col].map(
        lambda sparql_query:
        retrieve_all_kb_part(sparql_query, prefix_s, prefix_url_dict, fullfill_with_url=False)
    ).explode().drop_duplicates().dropna().map(
        lambda x: x if x.startswith(head_pattern) else np.nan
    ).dropna().map(
        lambda x: x[3:].strip()
    ).drop_duplicates()

    all_entity_info_df = pd.DataFrame(all_entity_s)
    all_entity_info_df.columns = ["entityid"]

    en_req = []
    zh_req = []
    for entityid in tqdm(all_entity_info_df["entityid"].tolist()):
        en_req.append(
        search_entity_rep_by_lang_filter(entityid, "en")
        )
        zh_req.append(
        search_entity_rep_by_lang_filter(entityid, "zh")
        )

    all_entity_info_df["en_info"] = en_req
    all_entity_info_df["zh_info"] = zh_req

    need_info_entityid_s = all_entity_info_df[
    all_entity_info_df.apply(
    lambda s: not s["en_info"] or not s["zh_info"]
    , axis = 1)
    ]["entityid"].drop_duplicates()
    return need_info_entityid_s
    #need_info_entityid_s.to_csv("need_info_entityid_s.csv", index = False)

class NerFix(object):
    @timer()
    def __init__(self, pool_size = 3):
        self.trans_model = EasyNMT('opus-mt')
        self.trans_dict = {}
        self.sim_model = SentenceTransformer('LaBSE')

        t = time()
        if pool_size > 0:
            pool0 = self.sim_model.start_multi_process_pool(['cpu'] * pool_size)
            print("start time consume :", time() - t)
        else:
            pool0 = None

        self.sim_model.pool = pool0

        t = time()
        if pool_size > 0:
            pool1 = self.trans_model.start_multi_process_pool(['cpu'] * pool_size)
            print("start time consume :", time() - t)
        else:
            pool1 = None

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

if __name__ == "__main__":
    '''
    #### if use emb and trans interface, should always run its django server.

    df = load_data()
    property_info_df = load_property_info_df()
    pid_relate_entity_df, pid_tuple_on_s_dict = load_pid_relate_entity_df()
    zh_entity_db = load_pid_tuple_on_s_dict_zh_entity_search_table(pid_tuple_on_s_dict)

    ner_fix = NerFix()

    #### en is the require
    #### err samples: df.iloc[5] df.iloc[7]
    #### property_info_df may always need some expand, such as add "is" to "P31"
    #### or code fix it.
    #### and lemmatize fix some sub representation

    #### df.iloc[13] other prefix of property (p:, ps:) seems also can aug
    #### take this new pattern in prefix_s in the future.
    #### en_sent : 'When position did Angela Merkel hold on November 10, 1994?'
    #### query : "SELECT ?obj WHERE { wd:Q567 p:P39 ?s . ?s ps:P39 ?obj . ?s pq:P580 ?x filter(contains(YEAR(?x),'1994')) }"

    #### performance problem. (cache \ pre_build en_zh_info dict \ data to db to search)

    #### entity property overlap drop by map_guess_dict_to_in_sent_mapping fuzz
    #### some entity or property string too long

    en_sent, sparql_query = df.iloc[0].tolist()

    a, b, c, d, e, f = sim_representation_decomp(en_sent, sparql_query,
                             prefix_s, prefix_url_dict,
                                             property_info_df,
                                             pid_tuple_on_s_dict,
                              zh_linker_entities,
                                ner_fix,
                                zh_entity_db,
                              only_guess = False,
                              entity_aug_size = 50
                             )

    guess_dict_in_sent_mapping_df_exploded = produce_sim_representation_reconstruct_df(
        en_sent, sparql_query, property_info_df,
        a, b, c, d, e
    )

    need_insert_p_list = seek_need_insert_property_list_from_df(guess_dict_in_sent_mapping_df_exploded)
    if need_insert_p_list:
        assert all(map(lambda p: p in a, need_insert_p_list))
    need_insert_p_list

    #### inter_str of property may be empty, because not in en_sent
    #### if can have a method to guess a insert ?

    #### if not in the simplest method to insert it, is in the head
    #### or in the tail directly.

    #### the property is in the context meaning.

    #### aug entity sim in property sense, may should also add literal some sim sense
    #### by search (bm25)

    #### may need add a sampler when retrieve_diff_query = False to "expand"

    aug_df = sim_representation_reconstruct_by_df(en_sent,
        sparql_query,
        guess_dict_in_sent_mapping_df_exploded[
    guess_dict_in_sent_mapping_df_exploded["inter_str"].map(lambda x: x.strip()).map(bool)
]
        , aug_times = 1000, retrieve_diff_query = False)
    aug_df

    if need_insert_p_list:
        aug_df_fixed = perform_all_insert_on_need_insert_p_list(
en_sent,
ner_fix,
property_info_df,
guess_dict_in_sent_mapping_df_exploded,
aug_df, need_insert_p_list,
)

    #### en_sent: 'which cola starts with the letter p'
    #### sparql_query: "SELECT DISTINCT ?sbj ?sbj_label WHERE { ?sbj wdt:P31 wd:Q134041 . ?sbj rdfs:label ?sbj_label . FILTER(STRSTARTS(lcase(?sbj_label), 'p')) . FILTER (lang(?sbj_label) = 'en') } LIMIT 25 "
    #### use perform_all_insert_on_need_insert_p_list without + ["is"]
    #### sent_trans: 'which cola is a starts with the letter p'
    #### use perform_all_insert_on_need_insert_p_list with + ["is"]
    #### sent_trans: 'which cola is starts with the letter p'

    if need_insert_p_list:
        ####
        aug_df_fixed = perform_all_insert_on_need_insert_p_list(
en_sent,
ner_fix,
property_info_df.apply(
    lambda x: pd.Series({
        "pid": x["pid"],
        "en_info": x["en_info"] + ["is"],
        "zh_info": x["zh_info"]
    }) if x["pid"] == "P31" else x
    , axis = 1),
guess_dict_in_sent_mapping_df_exploded,
aug_df, need_insert_p_list,
)

    #### query decomp
    df_cp = df.copy()
    l0 = []
    l1 = []
    for sparql_query in tqdm(df_cp["sparql_wikidata"].tolist()):
        ele0 = retrieve_all_kb_part(sparql_query, prefix_s, prefix_url_dict, fullfill_with_url=False)
        ele1 = retrieve_all_kb_part_wide(sparql_query, prefix_s)
        l0.append(ele0)
        l1.append(ele1)
    df_cp["all_kb_part"] = l0
    df_cp["all_kb_part_wide"] = l1

    #### not support query service query:
    #### SELECT ?answer WHERE { wd:Q118 wdt:P138 ?answer . ?answer wdt:P1843 wd:Quercia}
    #### wd:Quercia
    #### rdfs:label support
    df_cp[
    df_cp.apply(
        lambda x: set(x["all_kb_part"]).union(set(["rdfs:label"]))
        !=
        set(x["all_kb_part_wide"]).union(set(["rdfs:label"])), axis = 1
    )
]
    '''

    train_df = pd.read_json("train.json")
    test_df = pd.read_json("test.json")
    lcquad_2_0_df = pd.read_json("lcquad_2_0.json")
    df = pd.concat(list(map(lambda x: x[["question", "sparql_wikidata"]], [train_df, test_df, lcquad_2_0_df])), axis = 0).drop_duplicates()

    df = df.rename(
        columns = {
            "question": "en",
            "sparql_wikidata": "zh"
        }
    )[["en", "zh"]]


    property_info_df["en_info"] = property_info_df["info_dict"].map(
        lambda x: x.get("en", [])
    ).map(lambda x: map(clean_single_str, x)).map(list)
    property_info_df["zh_info"] = property_info_df["info_dict"].map(
        lambda x: x.get("zh", [])
    ).map(lambda x: map(clean_single_str, x)).map(list)

    property_info_df["pid"] = property_info_df["entities"].map(
        lambda x: list(x.keys())[0]
    )

    property_info_df = property_info_df[["pid", "en_info", "zh_info"]]

    x = 'SELECT ?answer WHERE { wd:Q169794 wdt:P26 ?X . ?X wdt:P22 ?answer}'

    retrieve_all_kb_part(x, prefix_s, prefix_url_dict, fullfill_with_url=True)
    '''
    ['http://www.wikidata.org/entity/Q169794',
     'http://www.wikidata.org/prop/direct/P26',
     'http://www.wikidata.org/prop/direct/P22']
    '''

    retrieve_all_kb_part(x, prefix_s, prefix_url_dict, fullfill_with_url=False)
    '''
    ['wd:Q169794', 'wdt:P26', 'wdt:P22']
    '''

    en_sent = "Who is the children of Ranavalona I's husbands?"
    entity_prop_dict = dict(property_info_df[
        property_info_df["pid"].isin(
            ["P22", "P26"]
        )
    ][["pid", "en_info"]].values.tolist())
    entity_prop_dict["Q169794"] = search_entity_rep_by_lang_filter("Q169794")

    guess_sim_representation(en_sent, entity_prop_dict)


    en_sent = "Who is the child of Ranavalona I's husband?"
    entity_prop_dict = dict(property_info_df[
        property_info_df["pid"].isin(
            ["P22", "P26"]
        )
    ][["pid", "en_info"]].values.tolist())
    entity_prop_dict["Q169794"] = search_entity_rep_by_lang_filter("Q169794")

    guess_sim_representation(en_sent, entity_prop_dict)

    #### consider fix property aug

    #### single sample aug
    #### property string product
    #### entity replacement

    #### aug cross id
    #### measure similarity cross entity and property

    pid_relate_entity_df = pd.read_json("lcquad_pid_relate_entity.json")
    #pid_relate_entity_df = pd.read_json("pid_relate_entity.json")
    pid_relate_entity_df["pid"] = pid_relate_entity_df["l"].map(
        lambda x: x["pid"]
    )
    pid_relate_entity_df["s"] = pid_relate_entity_df["l"].map(
        lambda x: x["s"]
    )
    pid_relate_entity_df = pid_relate_entity_df[["pid", "s"]]

    #### time consume
    pid_tuple_on_s_dict = produce_pid_tuple_on_s_dict(pid_relate_entity_df)

    some_entities = df["zh"].head(10).map(
        lambda y: retrieve_all_kb_part(y, prefix_s, prefix_url_dict, fullfill_with_url=False)
    ).explode().map(
        lambda x: x if x.startswith("wd:Q") else np.nan
    ).dropna().drop_duplicates().map(
        lambda x: x.strip()
    )

    all_entity_s = df["zh"].map(
        lambda sparql_query:
        retrieve_all_kb_part(sparql_query, prefix_s, prefix_url_dict, fullfill_with_url=False)
    ).explode().drop_duplicates().dropna().map(
        lambda x: x if x.startswith("wd:Q") else np.nan
    ).dropna().map(
        lambda x: x[3:].strip()
    ).drop_duplicates()

    all_entity_info_df = pd.DataFrame(all_entity_s)
    all_entity_info_df.columns = ["entityid"]

    en_req = []
    zh_req = []
    for entityid in tqdm(all_entity_info_df["entityid"].tolist()):
        en_req.append(
        search_entity_rep_by_lang_filter(entityid, "en")
        )
        zh_req.append(
        search_entity_rep_by_lang_filter(entityid, "zh")
        )

    all_entity_info_df["en_info"] = en_req
    all_entity_info_df["zh_info"] = zh_req

    need_info_entityid_s = all_entity_info_df[
    all_entity_info_df.apply(
    lambda s: not s["en_info"] or not s["zh_info"]
    , axis = 1)
    ]["entityid"].drop_duplicates()

    need_info_entityid_s.to_csv("need_info_entityid_s.csv", index = False)


    zh_en_df = pd.DataFrame(some_entities.map(
        lambda x: (x,
                   search_entity_rep_by_lang_filter(x[3:], "zh"),
                   search_entity_rep_by_lang_filter(x[3:], "en")
                  )
    ).values.tolist(), columns = ["entityid", "zh", "en"])

    ##### repaid
    zh_en_df["sim_entity_list"] = zh_en_df["entityid"].map(
        lambda x:
        search_sim_entity_by_property_count(x[3:], pid_relate_entity_df).head(100)["s"].tolist()
    )

    zh_en_df["sim_entity_list_f"] = zh_en_df["sim_entity_list"].map(
        lambda l: filter(lambda x: not x.startswith("维基") and not x.startswith("維基"), l)
    ).map(list)

    zh_en_df["sim_entity_list_f"] = zh_en_df.apply(
        lambda s:
        sorted(s["sim_entity_list_f"], key = lambda x:
               max(map(lambda y: fuzz.ratio(x, y), s["zh"]))
              , reverse = True)
        if s["zh"] else s["sim_entity_list_f"], axis = 1
    )

    #### some zh string entityid in "sim_entity_list_f"
    find_zh_str_entityid_by_linking("深紫樂隊", zh_linker_entities)

    search_entity_rep_by_lang_filter("Q101505", "en")

    search_entity_rep_by_lang_filter("Q101505", "zh")

    find_zh_str_entityid_by_linking("沃爾夫-隆達馬克-梅洛帝星系", zh_linker_entities)

    search_entity_rep_by_lang_filter("Q1155745", "en")

    search_entity_rep_by_lang_filter("Q1155745", "zh")

    find_zh_str_entityid_by_linking("亨利一世", zh_linker_entities)

    search_entity_rep_by_lang_filter("Q218190", "en")

    search_entity_rep_by_lang_filter("Q218190", "zh")

    ner_fix = NerFix()

    #### en is the require
    en_sent, sparql_query = df.iloc[0].tolist()

    a, b, c, d, e, f = sim_representation_decomp(en_sent, sparql_query,
                             prefix_s, prefix_url_dict,
                                             property_info_df,
                                             pid_tuple_on_s_dict,
                              zh_linker_entities,
                                ner_fix,
                              only_guess = False,
                              entity_aug_size = 50
                             )

    guess_dict_in_sent_mapping_df_exploded = produce_sim_representation_reconstruct_df(
        en_sent, sparql_query, property_info_df,
        a, b, c, d, e
    )

    need_insert_p_list = seek_need_insert_property_list_from_df(guess_dict_in_sent_mapping_df_exploded)
    if need_insert_p_list:
        assert all(map(lambda p: p in a, need_insert_p_list))

    #### inter_str of property may be empty, because not in en_sent
    #### if can have a method to guess a insert ?

    #### if not in the simplest method to insert it, is in the head
    #### or in the tail directly.

    #### the property is in the context meaning.

    #### aug entity sim in property sense, may should also add literal some sim sense
    #### by search (bm25)

    aug_df = sim_representation_reconstruct_by_df(en_sent,
        sparql_query, guess_dict_in_sent_mapping_df_exploded, aug_times = 1000)
    aug_df

    if need_insert_p_list:
        aug_df_fixed = perform_all_insert_on_need_insert_p_list(
    en_sent,
    ner_fix,
    property_info_df,
    guess_dict_in_sent_mapping_df_exploded,
    aug_df, need_insert_p_list,
)
