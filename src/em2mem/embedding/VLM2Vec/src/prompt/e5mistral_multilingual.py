from typing import Dict
from src.prompt.base_prompt import AutoPrompt


@AutoPrompt.register("e5mistral_multilingual")
def load_e5mistral_multilingual_prompt(task_name, task_type, *args, **kwargs):
    if not task_type:
        task_type = "Retrieval"
    if task_name.endswith("_small") or task_name.endswith("_s") or task_name.endswith("_xs"):
        task_name = task_name[:task_name.rindex("_")]
    if task_name.startswith("cqadupstack-"):
        task_name = "cqadupstack"
    task_def = get_task_def_by_task_name_and_type(task_name=task_name, task_type=task_type)
    prompt = get_detailed_instruct(task_def)
    prompt_dict = {"q_prompt": prompt, "d_prompt": ""}
    return prompt_dict


def get_task_def_by_task_name_and_type(task_type: str, task_name: str) -> str:
    # @ruimeng added
    if task_name.lower() in ['nli', 'allnli']:
        return "Retrieve a sentence that is semantically entailed by the given sentence."

    if task_type in ['STS', 'sts']:
        return "Retrieve semantically similar text."

    if task_type in ['Summarization', 'summarization']:
        return "Given a news summary, retrieve other semantically similar summaries"

    if task_type in ['BitextMining', 'bitextmining']:
        return "Retrieve parallel sentences."

    if task_type in ['Classification', 'classification']:
        task_name_to_instruct: Dict[str, str] = {
            'AmazonCounterfactualClassification': 'Classify a given Amazon customer review text as either counterfactual or not-counterfactual',
            'AmazonPolarityClassification': 'Classify Amazon reviews into positive or negative sentiment',
            'AmazonReviewsClassification': 'Classify the given Amazon review into its appropriate rating category',
            'Banking77Classification': 'Given a online banking query, find the corresponding intents',
            'EmotionClassification': 'Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise',
            'ImdbClassification': 'Classify the sentiment expressed in the given movie review text from the IMDB dataset',
            'MassiveIntentClassification': 'Given a user utterance as query, find the user intents',
            'MassiveScenarioClassification': 'Given a user utterance as query, find the user scenarios',
            'MTOPDomainClassification': 'Classify the intent domain of the given utterance in task-oriented conversation',
            'MTOPIntentClassification': 'Classify the intent of the given utterance in task-oriented conversation',
            'ToxicConversationsClassification': 'Classify the given comments as either toxic or not toxic',
            'TweetSentimentExtractionClassification': 'Classify the sentiment of a given tweet as either positive, negative, or neutral',
            # C-MTEB eval instructions
            'TNews': 'Classify the fine-grained category of the given news title',
            'IFlyTek': 'Given an App description text, find the appropriate fine-grained category',
            'MultilingualSentiment': 'Classify sentiment of the customer review into positive, neutral, or negative',
            'JDReview': 'Classify the customer review for iPhone on e-commerce platform into positive or negative',
            'OnlineShopping': 'Classify the customer review for online shopping into positive or negative',
            'Waimai': 'Classify the customer review from a food takeaway platform into positive or negative',
            # MTEB-pl eval instructions
            "CBD":"Classify the sentiment of polish tweet reviews",
            "PolEmo2.0-IN": "Classify the sentiment of in-domain (medicine and hotels) online reviews",
            "PolEmo2.0-OUT":"Classify the sentiment of out-of-domain (products and school) online reviews",
            "AllegroReviews": "Classify the sentiment of reviews from e-commerce marketplace Allegro",
            "PAC": "Classify the sentence into one of the two types: \"BEZPIECZNE_POSTANOWIENIE_UMOWNE\" and \"KLAUZULA_ABUZYWNA\"",
            # F-MTEB eval instructions
            "MasakhaNEWSClassification": "Classify the topic or theme of the given news articles based on the titles and contents",
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Clustering', 'clustering']:
        task_name_to_instruct: Dict[str, str] = {
            'ArxivClusteringP2P': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
            'ArxivClusteringS2S': 'Identify the main and secondary category of Arxiv papers based on the titles',
            'BiorxivClusteringP2P': 'Identify the main category of Biorxiv papers based on the titles and abstracts',
            'BiorxivClusteringS2S': 'Identify the main category of Biorxiv papers based on the titles',
            'MedrxivClusteringP2P': 'Identify the main category of Medrxiv papers based on the titles and abstracts',
            'MedrxivClusteringS2S': 'Identify the main category of Medrxiv papers based on the titles',
            'RedditClustering': 'Identify the topic or theme of Reddit posts based on the titles',
            'RedditClusteringP2P': 'Identify the topic or theme of Reddit posts based on the titles and posts',
            'StackExchangeClustering': 'Identify the topic or theme of StackExchange posts based on the titles',
            'StackExchangeClusteringP2P': 'Identify the topic or theme of StackExchange posts based on the given paragraphs',
            'TwentyNewsgroupsClustering': 'Identify the topic or theme of the given news articles',
            # C-MTEB eval instructions
            'CLSClusteringS2S': 'Identify the main category of scholar papers based on the titles',
            'CLSClusteringP2P': 'Identify the main category of scholar papers based on the titles and abstracts',
            'ThuNewsClusteringS2S': 'Identify the topic or theme of the given news articles based on the titles',
            'ThuNewsClusteringP2P': 'Identify the topic or theme of the given news articles based on the titles and contents',
            # MTEB-fr eval instructions
            "AlloProfClusteringP2P": "Identify the main category of Allo Prof document based on the titles and descriptions",
            "AlloProfClusteringS2S": "Identify the main category of Allo Prof document based on the titles",
            "HALClusteringS2S": "Identify the main category of academic passage based on the titles and contents",
            "MasakhaNEWSClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
            "MasakhaNEWSClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
            "MLSUMClusteringP2P": "Identify the topic or theme of the given articles based on the titles and contents",
            "MLSUMClusteringS2S":  "Identify the topic or theme of the given articles based on the titles",
            # MTEB-pl eval instructions
            "8TagsClustering": "Identify of headlines from social media posts in Polish  into 8 categories: film, history, food, medicine, motorization, work, sport and technology",
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Reranking', 'PairClassification', 'reranking', 'pairclassification']:
        task_name_to_instruct: Dict[str, str] = {
            'AskUbuntuDupQuestions': 'Retrieve duplicate questions from AskUbuntu forum',
            'MindSmallReranking': 'Retrieve relevant news articles based on user browsing history',
            'SciDocsRR': 'Given a title of a scientific paper, retrieve the titles of other relevant papers',
            'StackOverflowDupQuestions': 'Retrieve duplicate questions from StackOverflow forum',
            'SprintDuplicateQuestions': 'Retrieve duplicate questions from Sprint forum',
            'TwitterSemEval2015': 'Retrieve tweets that are semantically similar to the given tweet',
            'TwitterURLCorpus': 'Retrieve tweets that are semantically similar to the given tweet',
            # C-MTEB eval instructions
            'T2Reranking': 'Given a Chinese search query, retrieve web passages that answer the question',
            'MmarcoReranking': 'Given a Chinese search query, retrieve web passages that answer the question',
            'CMedQAv1': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'CMedQAv2': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'Ocnli': 'Retrieve semantically similar text.',
            'Cmnli': 'Retrieve semantically similar text.',
            # MTEB-fr eval instructions
            "AlloprofReranking": "Given a question, retrieve passages that answer the question",
            "OpusparcusPC":"Retrieve semantically similar text",
            "PawsX":"Retrieve semantically similar text",
            "SyntecReranking": "Given a question, retrieve passages that answer the question",
            # MTEB-pl eval instructions
            "SICK-E-PL": "Retrieve semantically similar text",
            "PPC": "Retrieve semantically similar text",
            "CDSC-E": "Retrieve semantically similar text",
            "PSC": "Retrieve semantically similar text",
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Retrieval', 'retrieval']:
        if task_name.lower().startswith('cqadupstack'):
            return 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question'

        task_name_to_instruct: Dict[str, str] = {
            'ArguAna': 'Given a claim, find documents that refute the claim',
            'ClimateFEVER': 'Given a claim about climate change, retrieve documents that support or refute the claim',
            'DBPedia': 'Given a query, retrieve relevant entity descriptions from DBPedia',
            'FEVER': 'Given a claim, retrieve documents that support or refute the claim',
            'FiQA2018': 'Given a financial question, retrieve user replies that best answer the question',
            'HotpotQA': 'Given a multi-hop question, retrieve documents that can help answer the question',
            'MSMARCO': 'Given a web search query, retrieve relevant passages that answer the query',
            'NFCorpus': 'Given a question, retrieve relevant documents that best answer the question',
            'NQ': 'Given a question, retrieve Wikipedia passages that answer the question',
            'QuoraRetrieval': 'Given a question, retrieve questions that are semantically equivalent to the given question',
            'SCIDOCS': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper',
            'SciFact': 'Given a scientific claim, retrieve documents that support or refute the claim',
            'Touche2020': 'Given a question, retrieve detailed and persuasive arguments that answer the question',
            'TRECCOVID': 'Given a query on COVID-19, retrieve documents that answer the query',
            # C-MTEB eval instructions
            'T2Retrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
            'MMarcoRetrieval': 'Given a web search query, retrieve relevant passages that answer the query',
            'DuRetrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
            'CovidRetrieval': 'Given a question on COVID-19, retrieve news articles that answer the question',
            'CmedqaRetrieval': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'EcomRetrieval': 'Given a user query from an e-commerce website, retrieve description sentences of relevant products',
            'MedicalRetrieval': 'Given a medical question, retrieve user replies that best answer the question',
            'VideoRetrieval': 'Given a video search query, retrieve the titles of relevant videos',
            # MTEB-fr eval instructions
            "AlloprofRetrieval": "Given a question, retrieve passages that answer the question",
            "BSARDRetrieval": "Given a question, retrieve passages that answer the question",
            "SyntecRetrieval": "Given a question, retrieve passages that answer the question",
            "XPQARetrieval": "Given a question, retrieve passages that answer the question",
            "MintakaRetrieval": "Given a question, retrieve passages that answer the question",
            # MTEB-pl eval instructions
            "ArguAna-PL": "Given a claim, find documents that refute the claim",
            "DBPedia-PL": "Given a query, retrieve relevant entity descriptions from DBPedia",
            "FiQA-PL": "Given a financial question, retrieve user replies that best answer the question",
            "HotpotQA-PL": "Given a multi-hop question, retrieve documents that can help answer the question",
            "MSMARCO-PL": "Given a web search query, retrieve relevant passages that answer the query",
            "NFCorpus-PL": "Given a question, retrieve relevant documents that best answer the question",
            "NQ-PL": "Given a question, retrieve Wikipedia passages that answer the question",
            "Quora-PL": "Given a question, retrieve questions that are semantically equivalent to the given question",
            "SCIDOCS-PL": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
            "SciFact-PL": "Given a scientific claim, retrieve documents that support or refute the claim",
            "TRECCOVID-PL": "Given a query on COVID-19, retrieve documents that answer the query"
        }

        # add lower case keys to match some beir names
        task_name_to_instruct.update({k.lower(): v for k, v in task_name_to_instruct.items()})
        # other cases where lower case match still doesn't work
        task_name_to_instruct['trec-covid'] = task_name_to_instruct['TRECCOVID']
        task_name_to_instruct['climate-fever'] = task_name_to_instruct['ClimateFEVER']
        task_name_to_instruct['dbpedia-entity'] = task_name_to_instruct['DBPedia']
        task_name_to_instruct['webis-touche2020'] = task_name_to_instruct['Touche2020']
        task_name_to_instruct['fiqa'] = task_name_to_instruct['FiQA2018']
        task_name_to_instruct['scidocs'] = task_name_to_instruct['SCIDOCS']
        task_name_to_instruct['quora'] = task_name_to_instruct['QuoraRetrieval']

        # for miracl evaluation
        task_name_to_instruct['miracl'] = 'Given a question, retrieve Wikipedia passages that answer the question'

        return task_name_to_instruct[task_name]

    raise ValueError(f"No instruction config for task_name `{task_name}` with task_type `{task_type}`")


def get_detailed_instruct(task_description: str) -> str:
    if not task_description:
        return ''

    return 'Instruct: {}\nQuery: '.format(task_description)