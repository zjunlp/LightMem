from src.prompt.base_prompt import AutoPrompt

@AutoPrompt.register("instructor")
def load_instructor_prompt(task_name, task_type="Retrieval", *args, **kwargs):
    if task_name.endswith('_small'):
        task_name = task_name[:-6]
    assert task_name in instructor_prompts_map, f'{task_name} is not supported in the InstructOR prompts'
    return instructor_prompts_map[task_name]


instructor_prompts_map = {
    # ST data
    'msmarco-triplets': {'q_prompt': 'Represent the question for retrieving supporting documents: ', 'd_prompt': 'Represent the document for retrieval: '},
    # added by Rui
    'AllNLI': {'q_prompt': 'Represent the sentence to retrieve semantically similar sentences: ', 'd_prompt': 'Represent the sentence to retrieve semantically similar sentences: '},
    'allnli': {'q_prompt': 'Represent the sentence to retrieve semantically similar sentences: ', 'd_prompt': 'Represent the sentence to retrieve semantically similar sentences: '},
    'bioasq': {'q_prompt': 'Represent a Science question for retrieving supporting papers: ', 'd_prompt': 'Represent the Public medical articles for retrieval: '},
    "signal1m": {"q_prompt": "Represent a news article title for retrieving relevant tweets", "d_prompt": "Represent the tweet (short text) for retrieval: "},
    "trec-news": {"q_prompt": "Represent a news headline to retrieve relevant news articles that provide background information", "d_prompt": "Represent the news article for retrieval: "},
    "robust04": {"q_prompt": "Represent a question to retrieve relevant news articles", "d_prompt": "Represent the news article for retrieval: "},

    # BEIR data
    'ClimateFEVER': {'q_prompt': 'Represent the Climate question for retrieving supporting documents: ', 'd_prompt': 'Represent the document for retrieval: '},
    'climate-fever': {'q_prompt': 'Represent the Climate question for retrieving supporting documents: ', 'd_prompt': 'Represent the document for retrieval: '},
    'HotpotQA': {'q_prompt': 'Represent the Wikipedia question for retrieving supporting documents: ', 'd_prompt': 'Represent the document for retrieval: '},
    'hotpotqa': {'q_prompt': 'Represent the Wikipedia question for retrieving supporting documents: ', 'd_prompt': 'Represent the document for retrieval: '},
    'FEVER': {'q_prompt': 'Represent the fact for retrieving supporting evidence: ', 'd_prompt': 'Represent the evidence for retrieval: '},
    'fever': {'q_prompt': 'Represent the fact for retrieving supporting evidence: ', 'd_prompt': 'Represent the evidence for retrieval: '},
    'MSMARCO': {'q_prompt': 'Represent the question for retrieving supporting documents: ', 'd_prompt': 'Represent the document for retrieval: '},
    'msmarco': {'q_prompt': 'Represent the question for retrieving supporting documents: ', 'd_prompt': 'Represent the document for retrieval: '},
    'DBPedia': {'q_prompt': 'Represent the Wikipedia questions to retrieve a supporting document: ', 'd_prompt': 'Represent the Wikipedia documents for retrieval: '},
    'dbpedia-entity': {'q_prompt': 'Represent the Wikipedia questions to retrieve a supporting document: ', 'd_prompt': 'Represent the Wikipedia documents for retrieval: '},
    'NQ': {'q_prompt': 'Represent the Wikipedia question for retrieving supporting documents: ', 'd_prompt': 'Represent the document for retrieval: '},
    'nq': {'q_prompt': 'Represent the Wikipedia question for retrieving supporting documents: ', 'd_prompt': 'Represent the document for retrieval: '},
    'QuoraRetrieval': {'q_prompt': 'Represent the Quora question to retrieve question: ', 'd_prompt': 'Represent the Quora question to retrieve question: '},
    'quora': {'q_prompt': 'Represent the Quora question to retrieve question: ', 'd_prompt': 'Represent the Quora question to retrieve question: '},
    'SCIDOCS': {'q_prompt': 'Represent a Science question for retrieving supporting papers: ', 'd_prompt': 'Represent the Science paper: '},
    'scidocs': {'q_prompt': 'Represent a Science question for retrieving supporting papers: ', 'd_prompt': 'Represent the Science paper: '},
    'TRECCOVID': {'q_prompt': 'Represent the Coronavirus questions to retrieve a supporting document: ', 'd_prompt': 'Represent the Coronavirus documents for retrieval: '},
    'trec-covid': {'q_prompt': 'Represent the Coronavirus questions to retrieve a supporting document: ', 'd_prompt': 'Represent the Coronavirus documents for retrieval: '},
    'Touche2020': {'q_prompt': 'Represent questions: ', 'd_prompt': 'Represent arguments: '},
    'webis-touche2020': {'q_prompt': 'Represent questions: ', 'd_prompt': 'Represent arguments: '},
    'SciFact': {'q_prompt': 'Represent the Scientific queries for retrieving a supporting passage: ', 'd_prompt': 'represent the scientific paragraph for retrieval: '},
    'scifact': {'q_prompt': 'Represent the Scientific queries for retrieving a supporting passage: ', 'd_prompt': 'represent the scientific paragraph for retrieval: '},
    'NFCorpus': {'q_prompt': 'Represent the nutrition facts to retrieve Public medical articles: ', 'd_prompt': 'Represent the Public medical articles for retrieval: '},
    'nfcorpus': {'q_prompt': 'Represent the nutrition facts to retrieve Public medical articles: ', 'd_prompt': 'Represent the Public medical articles for retrieval: '},
    'ArguAna': {'q_prompt': 'Represent Debating conversations to retrieve a counter-argument: ', 'd_prompt': 'Represent counter-arguments: '},
    'arguana': {'q_prompt': 'Represent Debating conversations to retrieve a counter-argument: ', 'd_prompt': 'Represent counter-arguments: '},
    'cqadupstack': {'q_prompt': '', 'd_prompt': ''},  # needs to be set inside the cqadupstack subset inner loop
    'CQADupstackTexRetrieval': {'q_prompt': 'Represent the question for retrieving answers: ', 'd_prompt': 'Represent the answer for retrieval: '},
    'cqadupstack-tex': {'q_prompt': 'Represent the question for retrieving answers: ', 'd_prompt': 'Represent the answer for retrieval: '},
    'CQADupstackWebmastersRetrieval': {'q_prompt': 'Represent the Webmaster question for retrieving answers: ', 'd_prompt': 'Represent the Webmaster answer: '},
    'cqadupstack-webmasters': {'q_prompt': 'Represent the Webmaster question for retrieving answers: ', 'd_prompt': 'Represent the Webmaster answer: '},
    'CQADupstackEnglishRetrieval': {'q_prompt': 'Represent the English question for retrieving documents: ', 'd_prompt': 'Represent the English answer for retrieval: '},
    'cqadupstack-english': {'q_prompt': 'Represent the English question for retrieving documents: ', 'd_prompt': 'Represent the English answer for retrieval: '},
    'CQADupstackGamingRetrieval': {'q_prompt': 'Represent the Gaming question for retrieving answers: ', 'd_prompt': 'Represent the Gaming answer for retrieval: '},
    'cqadupstack-gaming': {'q_prompt': 'Represent the Gaming question for retrieving answers: ', 'd_prompt': 'Represent the Gaming answer for retrieval: '},
    'CQADupstackGisRetrieval': {'q_prompt': 'Represent the Gis question for retrieving answers: ', 'd_prompt': 'Represent the Gis answer for retrieval: '},
    'cqadupstack-gis': {'q_prompt': 'Represent the Gis question for retrieving answers: ', 'd_prompt': 'Represent the Gis answer for retrieval: '},
    'CQADupstackUnixRetrieval': {'q_prompt': 'Represent the Unix questions to retrieve a supporting answer: ', 'd_prompt': 'Represent the Unix answers for retrieval: '},
    'cqadupstack-unix': {'q_prompt': 'Represent the Unix questions to retrieve a supporting answer: ', 'd_prompt': 'Represent the Unix answers for retrieval: '},
    'CQADupstackMathematicaRetrieval': {'q_prompt': 'Represent the Mathematical question for retrieving answers: ', 'd_prompt': 'Represent the Mathematical answer for retrieval: '},
    'cqadupstack-mathematica': {'q_prompt': 'Represent the Mathematical question for retrieving answers: ', 'd_prompt': 'Represent the Mathematical answer for retrieval: '},
    'CQADupstackStatsRetrieval': {'q_prompt': 'Represent the Statistical question for retrieving answers: ', 'd_prompt': 'Represent the Statistical answer for retrieval: '},
    'cqadupstack-stats': {'q_prompt': 'Represent the Statistical question for retrieving answers: ', 'd_prompt': 'Represent the Statistical answer for retrieval: '},
    'CQADupstackPhysicsRetrieval': {'q_prompt': 'Represent the Physics question for retrieving answers: ', 'd_prompt': 'Represent the Physics answer for retrieval: '},
    'cqadupstack-physics': {'q_prompt': 'Represent the Physics question for retrieving answers: ', 'd_prompt': 'Represent the Physics answer for retrieval: '},
    'CQADupstackProgrammersRetrieval': {'q_prompt': 'Represent the Programming question for retrieving answers: ', 'd_prompt': 'Represent the Programming answer for retrieval: '},
    'cqadupstack-programmers': {'q_prompt': 'Represent the Programming question for retrieving answers: ', 'd_prompt': 'Represent the Programming answer for retrieval: '},
    'CQADupstackAndroidRetrieval': {'q_prompt': 'Represent the Android question for retrieving answers: ', 'd_prompt': 'Represent the Android answer for retrieval: '},
    'cqadupstack-android': {'q_prompt': 'Represent the Android question for retrieving answers: ', 'd_prompt': 'Represent the Android answer for retrieval: '},
    'CQADupstackWordpressRetrieval': {'q_prompt': 'Represent the Wordpress question for retrieving answers: ', 'd_prompt': 'Represent the Wordpress answer for retrieval: '},
    'cqadupstack-wordpress': {'q_prompt': 'Represent the Wordpress question for retrieving answers: ', 'd_prompt': 'Represent the Wordpress answer for retrieval: '},
    'FiQA2018': {'q_prompt': 'Represent the finance questions to retrieve a supporting answer: ', 'd_prompt': 'Represent the finance answers for retrieval: '},
    'fiqa': {'q_prompt': 'Represent the finance questions to retrieve a supporting answer: ', 'd_prompt': 'Represent the finance answers for retrieval: '},
}
