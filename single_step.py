'''
####!pip install jionlp
####!pip install easynmt
####!pip install seaborn
####!pip install cn2an
####!pip install opencc
####!pip install nltk ##### import nltk;nltk.download("wordnet")

#### files:
/temp/kbqa-explore/wikidata.hdt
/temp/kbqa-explore/linker_entities.pkl
'''

import logging, sys
logging.disable(sys.maxsize)
from lcquad_query_aug_script_with_time import *

#### load compoents
template_df_in_deeppavlov = load_pickle("lcquad_in_deeppavlov_template_abstract.pkl")
property_info_df = load_property_info_df()
pid_relate_entity_df, pid_tuple_on_s_dict = load_pid_relate_entity_df()
zh_entity_db = load_pid_tuple_on_s_dict_zh_entity_search_table(pid_tuple_on_s_dict)
ner_fix = NerFix(pool_size = 0)
####

def aug_one_query(en_sent, sparql_query, aug_times = 1000):
    a, b, c, d, e, f = sim_representation_decomp(en_sent, sparql_query,
                             prefix_s, prefix_url_dict,
                                             property_info_df,
                                             pid_tuple_on_s_dict,
                              zh_linker_entities,
                                ner_fix,
                                zh_entity_db,
                              only_guess = False,
                              entity_aug_size = 50,
                              skip_no_db_zh_entity_str = True
                             )
    guess_dict_in_sent_mapping_df_exploded = produce_sim_representation_reconstruct_df(
        en_sent, sparql_query, property_info_df,
        a, b, c, d, e
    )
    aug_df = sim_representation_reconstruct_by_df(en_sent,
        sparql_query,
        guess_dict_in_sent_mapping_df_exploded[
        guess_dict_in_sent_mapping_df_exploded["inter_str"].map(lambda x: x.strip()).map(bool)
]
            , aug_times = aug_times, retrieve_diff_query = False)
    return aug_df

if __name__ == "__main__":
    #### try two lcquad query aug examles.
    en_sent = "What is ChemSpider ID of tungsten carbide ?"
    sparql_query = "select distinct ?answer where { wd:Q423265 wdt:P661 ?answer}"

    en_sent = "Name the women's association football team who play the least in tournaments."
    sparql_query = 'select ?ent where { ?ent wdt:P31 wd:Q1478437 . ?ent wdt:P2257 ?obj . ?ent wdt:P2094 wd:Q606060. } ORDER BY ASC(?obj)LIMIT 5 '

    aug_one_query(en_sent, sparql_query, aug_times=1000)
