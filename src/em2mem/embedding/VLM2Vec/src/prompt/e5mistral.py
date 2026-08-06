from typing import Dict
from src.prompt.base_prompt import AutoPrompt


@AutoPrompt.register("e5mistral")
def load_e5mistral_prompt(task_name, task_type, *args, **kwargs):
    if not task_type:
        task_type = "Retrieval"
    if task_name.endswith("_small") or task_name.endswith("_s") or task_name.endswith("_xs"):
        task_name = task_name[:task_name.rindex("_")]
    if task_name.endswith("_bgelarge"):
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
            'AmazonReviewsPairClassification': 'Given an Amazon review, locate reviews within the same rating category',
            'Banking77Classification': 'Given a online banking query, find the corresponding intents',
            'EmotionClassification': 'Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise',
            'EmotionPairClassification': 'Given an Twitter message, locate message within the same emotion category',
            'ImdbClassification': 'Classify the sentiment expressed in the given movie review text from the IMDB dataset',
            'MassiveIntentClassification': 'Given a user utterance as query, find the user intents',
            'MassiveScenarioClassification': 'Given a user utterance as query, find the user scenarios',
            'MTOPDomainClassification': 'Classify the intent domain of the given utterance in task-oriented conversation',
            'MTOPIntentClassification': 'Classify the intent of the given utterance in task-oriented conversation',
            'MTOPIntentPairClassification': 'Given an utterance in task-oriented conversation, locate utterance within the same intent category',
            'ToxicConversationsClassification': 'Classify the given comments as either toxic or not toxic',
            'ToxicConversationsPairClassification': 'Given an comment as toxic or non-toxic, locate comments within the same category',
            'TweetSentimentExtractionClassification': 'Classify the sentiment of a given tweet as either positive, negative, or neutral',
            'TweetSentimentPairClassification': 'Given an comment as either positive, negative, or neutral, locate comments within the same category',
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
            'InstructConversation': "Given a question asked by user, the assistant answers",
            'MrTydi': "Given a question, retrieve Wikipedia passages that answer the question",
            "ChatgptShortLong": "Given a query, retrieve passages that answer the query",
            # E5 public training
            "msmarco_document": "Given a web search query, retrieve relevant documents that answer the query",
            "msmarco_passage": "Given a web search query, retrieve relevant passages that answer the query",
            "allnli": "Given a web search query, retrieve relevant documents that answer the query",
            "dureader": "Given a Chinese search query, retrieve web passages that answer the question",
            "eli5_question_answer": "Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum",
            "fever": "Given a claim, retrieve documents that support or refute the claim",
            "hotpot_qa": "Given a multi-hop question, retrieve documents that can help answer the question",
            "miracl": "Given a question, retrieve Wikipedia passages that answer the question",
            "mrtydi": "Given a question, retrieve Wikipedia passages that answer the question",
            "nq": "Given a question, retrieve Wikipedia passages that answer the question",
            "quora_duplicates": "Given a question, retrieve questions that are semantically equivalent to the given question",
            "squad": "Retrieve Wikipedia passages that answer the question",
            "t2ranking": "Given a Chinese search query, retrieve web passages that answer the question",
            "trivia_qa": "Retrieve Wikipedia passages that answer the question",
            # code
            "codesearchnet": "Retrieve a code snippet that answers the given question",
            "humaneval": "Retrieve a code snippet that correctly solves the given programming problem",
            "mbpp": "Retrieve a code snippet that correctly solves the given programming problem",
            "ds1000": "Retrieve a library doc that answers the given question",
            "odex": "Retrieve a library doc that answers the given question",
            "xcodeeval_code2code": "Given a code snippet, retrieve a similar code snippet that is relevant",
            "xcodeeval_nl2code": "Given a natural language coding question, retrieve a relevant code snippet that answers the question",
            "xcodeeval_code2nl": "Given a code snippet, retrieve a relevant natural language snippet that describe the code",
            "swe-bench": "Given a issue text, retrieve the relevant code file",
            "repoeval": "Given a code snippet, retrieve the relevant code to it",
            "codestackexchange": "Given a code-related question, retrieve the relevant answer",
            "codesearchnet-ccr": "Given Code or Text, retrieval relevant content",
            "coir-t2c": "Given a text description or question, retrieve the complete code program that solves it",
            "coir-c2t": "Given a code snippet, retrieve the text that describes it",
            "coir-c2c": "Given a code snippet, retrieve relevant code to complete or enhance it",
            "coir-hybrid": "Given a code snippet and a related text description or question, retrieve the relevant code or text that completes, explains, or enhances it",
            # domain specific
            "trailhead": "Given a search query, retrieve relevant documents that answer the query",
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
        task_name_to_instruct['instructed-conversation'] = task_name_to_instruct['InstructConversation']
        task_name_to_instruct['berri'] = ""

        # for miracl evaluation
        task_name_to_instruct['miracl'] = 'Given a question, retrieve Wikipedia passages that answer the question'

        return task_name_to_instruct[task_name]

    raise ValueError(f"No instruction config for task_name `{task_name}` with task_type `{task_type}`")


def get_detailed_instruct(task_description: str) -> str:
    if not task_description:
        return ''

    return 'Instruct: {}\nQuery: '.format(task_description)