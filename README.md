<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">LC-QuAD-augmentation-toolkit</h3>

  <p align="center">
   		A augmentation toolkit with the help of DeepPavlov's wikidata tools
    <br />
  </p>
</p>

### Brief introduction
LC-QuAD 2.0 is a Large Question Answering dataset with 30,000 pairs of question and its corresponding SPARQL query. whose target knowledge base is Wikidata and DBpedia.<br/>
When construct a Knowledge Base Query system. The core of natural language to SPARQL query, may construct with the help of this dataset.If you are interested in this topic, i recommend you to read a brief introduction about [How to construct a Knowledge Base Question Answering system with the help of DeepPavlov in your language domain --- non English condition.](https://github.com/svjack/DeepPavlov-Chinese-KBQA/blob/main/design_construction.md)

This project is target on take one record in LC-QuAD dataset as input.(i.e. one English query and its corresponding SPARQL query as input pair), And the project will give you some similar self-construct sentence-sparql_query pairs as output in English.

Use these massive pairs and a translation toolkit, you can construct a KBQA system with the help of any sentence to query based Knowledge Base Engine. (as [DeepPavlov](https://github.com/deeppavlov/DeepPavlov)'s module do)

### Installation
Refer to [INSTALL.sh](INSTALL.sh) to install the environment, make sure that you can run the KBQA of the original DeepPavlov project.
The wikidata Knowledge Base hdt file can get from me or the [rdfhdt](https://www.rdfhdt.org/datasets/)

Below files should be located in the root path of this project after clone from repository.
```yml
lcquad_2_0.json
test.json
train.json

lcquad_in_deeppavlov_template_abstract.pkl
pid_tuple_on_s_dict.pkl
property_info_df.pkl

pid_tuple_on_s_dict.db
```
And below three files are too big to upload. 
<!--
You can email me to get them. (ehangzhou@outlook.com or svjackbt@gmail.com)
-->
You can use below link to get them from Baidu Yun Drive. And placed them in the project root path.
https://pan.baidu.com/s/1e66Lt6nisM3583dbIGsO5w?pwd=ntwz
<br/>
Remember use cat to merge wikidata.hdt.aa wikidata.hdt.ab wikidata.hdt.ac into wikidata.hdt to use it
```yml
multi_lang_kb_dict.db
kbqa-explore/wikidata.hdt
kbqa-explore/linker_entities.pkl
```

### Toolkit Usage
After environment installed, you can take a look at the snippet located in [single_step.py](https://github.com/svjack/LC-QuAD-augmentation-toolkit/blob/main/single_step.py).<br/>
It takes en_sent and sparql_query as input parameters and give a output in the format of pandas dataframe. Let's look at some examples that only
sample 5 outputs (the total populations may  from 3 to 1000).

<b>Example 1:</b>
```python
en_sent = "What is ChemSpider ID of tungsten carbide ?"
sparql_query = "select distinct ?answer where { wd:Q423265 wdt:P661 ?answer}"

np.random.seed(0)
df = aug_one_query(en_sent, sparql_query, aug_times=1000)
df = df.sample(n = 5).sort_values(by = "fuzz", ascending = False)
df.apply(lambda x: x.to_dict(), axis = 1).values.tolist()
```
This will output:
```python
[{'aug_en_sent': 'What is ChemSpider ID of tungsten trioxide ?',
  'aug_sparql_query': 'select distinct ?answer where { wd:Q417406 wdt:P661 ?answer}',
  'fuzz': 91.95402298850574},
 {'aug_en_sent': 'What is ChemSpider ID of hafnium(IV) carbide ?',
  'aug_sparql_query': 'select distinct ?answer where { wd:Q418001 wdt:P661 ?answer}',
  'fuzz': 80.89887640449437},
 {'aug_en_sent': 'What is ChemSpider ID of Carbonization ?',
  'aug_sparql_query': 'select distinct ?answer where { wd:Q2630655 wdt:P661 ?answer}',
  'fuzz': 74.69879518072288},
 {'aug_en_sent': 'What is identifier in a free chemical database, owned by the Royal Society of Chemistry of tungsten trioxide ?',
  'aug_sparql_query': 'select distinct ?answer where { wd:Q417406 wdt:P661 ?answer}',
  'fuzz': 44.44444444444444},
 {'aug_en_sent': 'What is identifier in a free chemical database, owned by the Royal Society of Chemistry of tantalum hafnium carbide ?',
  'aug_sparql_query': 'select distinct ?answer where { wd:Q424268 wdt:P661 ?answer}',
  'fuzz': 41.25}]
```

<b>Example 2:</b>
```python
en_sent = "Name the women's association football team who play the least in tournaments."
sparql_query = 'select ?ent where { ?ent wdt:P31 wd:Q1478437 . ?ent wdt:P2257 ?obj . ?ent wdt:P2094 wd:Q606060. } ORDER BY ASC(?obj)LIMIT 5 '

np.random.seed(0)
df = aug_one_query(en_sent, sparql_query, aug_times=1000)
df = df.sample(n = 5).sort_values(by = "fuzz", ascending = False)
df.apply(lambda x: x.to_dict(), axis = 1).values.tolist()
```
This will output:
```python
[{'aug_en_sent': 'Name the VfL Bochum team who compclass the least in tournaments.',
  'aug_sparql_query': 'select ?ent where { ?ent wdt:P31 wd:Q1478437 . ?ent wdt:P2257 ?obj . ?ent wdt:P2094 wd:Q105861. } ORDER BY ASC(?obj)LIMIT 5 ',
  'fuzz': 72.34042553191489},
 {'aug_en_sent': 'Name the VfL Wolfsburg team who competition class the least in tournaments.',
  'aug_sparql_query': 'select ?ent where { ?ent wdt:P31 wd:Q1478437 . ?ent wdt:P2257 ?obj . ?ent wdt:P2094 wd:Q101859. } ORDER BY ASC(?obj)LIMIT 5 ',
  'fuzz': 68.42105263157895},
 {'aug_en_sent': 'Name the Irapuato FC team who class for competition the least in tournaments.',
  'aug_sparql_query': 'select ?ent where { ?ent wdt:P31 wd:Q1478437 . ?ent wdt:P2257 ?obj . ?ent wdt:P2094 wd:Q1023193. } ORDER BY ASC(?obj)LIMIT 5 ',
  'fuzz': 67.53246753246754},
 {'aug_en_sent': 'Name the 1994 FIFA World Cup team who competition class the least in tournaments.',
  'aug_sparql_query': 'select ?ent where { ?ent wdt:P31 wd:Q1478437 . ?ent wdt:P2257 ?obj . ?ent wdt:P2094 wd:Q101751. } ORDER BY ASC(?obj)LIMIT 5 ',
  'fuzz': 65.82278481012658},
 {'aug_en_sent': 'Name the VfL Wolfsburg team who official classification by a regulating body under which the subject qualifies for inclusion the least in tournaments.',
  'aug_sparql_query': 'select ?ent where { ?ent wdt:P31 wd:Q1478437 . ?ent wdt:P2257 ?obj . ?ent wdt:P2094 wd:Q101859. } ORDER BY ASC(?obj)LIMIT 5 ',
  'fuzz': 51.98237885462555}]
```

<br/>
<h3>
<b>
Recommend you to read below parts:
</b>
</h3>

<!--
<h4>
<p>
<a href="design_construction.md"> Design Construction </a>
</p>
</h4>
This will give you a project summary.
-->

<h4>
<p>
<a href="api_doc.md"> API Documentation </a>
</p>
</h4>
This will help you have a knowledge of the detail function definition.

<h4>
<p>
<a href="https://github.com/svjack/DeepPavlov-Chinese-KBQA"> DeepPavlov-Chinese-KBQA </a>
</p>
</h4>

This will give you a demo about how to construct a KBQA system on a non-English language (take Chinese for example) with the help of [DeepPavlov](https://github.com/deeppavlov/DeepPavlov).

<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - svjackbt@gmail.com - ehangzhou@outlook.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link:[https://github.com/svjack/LC-QuAD-augmentation-toolkit](https://github.com/svjack/LC-QuAD-augmentation-toolkit)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
<!--
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)
-->
* [DeepPavlov](https://github.com/deeppavlov/DeepPavlov)
* [EasyNMT](https://github.com/UKPLab/EasyNMT)
* [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
* [rdfhdt](https://www.rdfhdt.org/datasets/)
* [rdflib](https://github.com/RDFLib/rdflib)
* [DeepPavlov-Chinese-KBQA](https://github.com/svjack/DeepPavlov-Chinese-KBQA)
* [tableQA-Chinese](https://github.com/svjack/tableQA-Chinese)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
