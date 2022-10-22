### Function Documentation
<b>lcquad_query_aug_script_with_time.py</b><br/>
Main script that perform the augmentation on lcquad dataset. The problem can be portrayed as decompose the SPARQL query first and find wikidataId from query. then find the representation of wikidataId corresponding entity in the input Englsh query, and find a similar entity with its id from wikidata Knowledge Base. and render them back into template of input English sentence and SPARQL query.

#### function definitions:
<b>load_data</b><br/>
<b>load_property_info_df</b><br/>
<b>load_pid_relate_entity_df</b><br/>
<b>produce_data_dict_for_search</b>:<br/>
load some datasource required, mainly about the lcquad dataset, propertis of wikidata in English and the entities to properties mappings (i.e. the graph node and its edge in wikidata Knowledge Graph)

<b>http_get_wiki_entity_property_info_by_id</b><br/>
<b>info_extracter:</b><br/>
Get info of entities and properties use request.

<b>retrieve_all_kb_part</b><br/>
<b>retrieve_all_kb_part_wide</b>:<br/>
decompose the SPARQL query and get all entityid and propertyid with the prefix as "wd" or "wdt"

<b>find_query_direction_format</b><br/>
<b>find_query_prop_format</b>:<br/>
forward or backward sparql query format runing on hdt file.

<b>one_part_g_producer</b><br/>
<b>py_dumpNtriple</b><br/>
<b>search_triples_with_parse:</b><br/>
look at https://github.com/svjack/DeepPavlov-Chinese-KBQA/blob/main/api_doc.md

<b>entity_property_search</b><br/>
generate data for function load_pid_relate_entity_df

<b>merge_nest_list</b><br/>
<b>get_match_blk_by_diff</b><br/>
<b>get_match_blk</b><br/>
<b>get_match_intersection</b><br/>
<b>get_match_intersection_fill_gap</b><br/>
<b>sent_list_match_to_df</b><br/>
<b>sent_list_match_to_df_with_bnd</b><br/>
<b>sent_list_match_to_df_bnd_cat:</b><br/>
Find a similar substring from a sentence and one string

<b>lemmatize_one_token</b><br/>
<b>lemma_score_match_it:</b><br/>
simplify treat for English.

<b>guess_sim_representation</b><br/>
<b>guess_sim_representation_by_score</b><br/>
<b>map_reduce_guess_sim_representation_by_score</b><br/>
Use Decompsition wikidataId find the most similar entity representation from wikidata Knowledge Base and retrieve its substring representation in English sentence.

<b>search_entity_rep_by_lang_filter</b><br/>
<b>search_entity_rep_by_lang_filter_in_db</b><br/>
<b>search_entity_rep_by_lang_filter_by_init_dict</b><br/>
look at https://github.com/svjack/DeepPavlov-Chinese-KBQA/blob/main/api_doc.md

<b>find_zh_str_entityid_by_linking</b><br/>
<b>find_zh_str_entityid_by_db</b><br/>
<b>find_en_str_entityid_by_trans_near_linking</b><br/>
Find the entityid (wikidataId) of a text representation, if the text not in Knowledge Base, use find_en_str_entityid_by_trans_near_linking to first find a "near" text (in the sense of embedding distance of [LaBSE](https://huggingface.co/sentence-transformers/LaBSE)) in Knowledge Base that have entityid and use them.

<b>produce_pid_tuple_on_s_dict</b><br/>
<b>load_pid_tuple_on_s_dict_zh_entity_search_table</b><br/>
generate data for propertis of wikidata in English and the entities to properties mappings.

<b>search_sim_entity_by_property_count_by_dict</b><br/>
<b>search_sim_entity_by_property_count_by_dict_add_fuzz</b><br/>
<b>search_sim_entity_by_property_count_by_dict_add_fuzz_f_by_db</b><br/>
<b>search_sim_entity_by_property_count</b><br/>
<b>search_sim_entity_by_property_count_by_explode</b><br/>
search similar entity on the count number of propertis in many self-prebuild different collections.

<b>sim_representation_decomp</b><br/>
main function to decompose English sentence and its corresponding SPARQL query. Find the mapping relation between wikidataId and corresponding natural language representation.

<b>phrase_validation</b><br/>
justify a span of sentence is or not a phrase in the sentence. define phrase as some tokens joined by some blanks.

<b>most_sim_token_in_sent</b><br/>
<b>recurrent_decomp_entity_str_by_en_sent</b><br/>
most_sim_token_in_sent is to find the most similar token in the sentence compared with a entity string. And recurrent_decomp_entity_str_by_en_sent use most_sim_token_in_sent in a recurrent way, that may find the longest phrase that most similar with the entity string.

<b>sp_string_by_desc_str_list</b><br/>
split the sentence by the description vocabs, the description vocabs defined like [gensim](https://radimrehurek.com/gensim/models/phrases.html)'s ENGLISH_CONNECTOR_WORDS
```python
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS
```
<br/>
<b>guess_most_sim_pharse_in_en_sent</b><br/>
<b>map_guess_dict_to_in_sent_mapping</b><br/>
<b>slice_guess_dict_in_sent_mapping_df</b><br/>
<b>map_guess_dict_to_in_sent_mapping_multi_times</b><br/>
the natural language representation in sim_representation_decomp may not a phrase but only a representation retrieve by  map_reduce_guess_sim_representation_by_score, so these functions transform the representation to a phrase from English sentence. These functions can be seen as a map from substring may be unreasonable into a reasonable phrase.
<br/><br/>

<b>produce_sim_representation_reconstruct_df</b><br/>
construct a dataframe whose different rows as entity representations and its meta data.(wikidataId, similar score and so on.)

<b>maintain_entity_cut_on_en_sent</b><br/>
<b>one_row_aug</b><br/>
augmentation perform function on above dataframe.

<b>sim_representation_reconstruct_by_df</b><br/>
main function use sim_representation_decomp to decompose and produce_sim_representation_reconstruct_df to reconstruct. Then we finished the whole augmentation task.

<b>only_fix_script_ser.py</b><br/>
<b>trans_emb_utils.py:</b><br/>
Some toolkits for translation and embedding produce with compare.
