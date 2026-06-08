import copy
from src.prompt.base_prompt import AutoPrompt

@AutoPrompt.register("tart")
def load_tart_prompt(task_name, task_type="Retrieval", *args, **kwargs):
    if task_name.endswith('_small'):
        task_name = task_name[:-6]
    if task_name.lower().startswith('cqadupstack'):
        prompt_dict = copy.copy(tart_prompts_map['cqadupstack'])
    else:
        assert task_name in tart_prompts_map, f'{task_name} is not supported in the TART prompts'
        prompt_dict = copy.copy(tart_prompts_map[task_name.lower()])
    SEP = ' [SEP] '  # used in BERRI
    prompt_dict['q_prompt'] += f'{SEP}'
    prompt_dict['d_prompt'] += f'{SEP}'
    return prompt_dict


tart_prompts_map = {
    # added by Rui
    'AllNLI': {'q_prompt': 'Represent the sentence to retrieve semantically similar sentences: ',
               'd_prompt': 'Represent the sentence to retrieve semantically similar sentences: '},

    # instructions used for BEIR evaluations (TART paper table 9: https://arxiv.org/pdf/2211.09260.pdf)
    "trec-covid": {"q_prompt": "Retrieve Scientific paper paragraph to answer this question", "d_prompt": "pubmed abstract related to COVID19"},
    "nfcorpus": {"q_prompt": "Retrieve Scientific paper paragraph to answer this question", "d_prompt": "pubmed abstract"},
    "fiqa": {"q_prompt": "Find financial web article paragraph to answer", "d_prompt": "Investment related StackExchange"},
    "arguana": {"q_prompt": "Retrieve an argument that counter argues the following paragraph", "d_prompt": "argument"},
    "webis-touche2020": {"q_prompt": "You have to retrieve an argument to this debate question", "d_prompt": "argument paragraph"},
    "dbpedia-entity": {"q_prompt": "Retrieve a Wikipedia introduction paragraph of the following entity", "d_prompt": "wikipedia paragraph"},
    "scidocs": {"q_prompt": "Find scientific paper titles that are related to the following", "d_prompt": "scientific paper abstract"},
    "climate-fever": {"q_prompt": "I want to know if the following claim is true or not. Retrieve a Wikipedia paragraph on climate change for this", "d_prompt": "wikipedia paragraph"},
    "scifact": {"q_prompt": "Retrieve a scientific paper sentence to verify if the following claim is true", "d_prompt": "scientific paper abstract"},

    # BEIR instructions from training (msmarco, NQ, HotpotQA, fever, quora)
    "msmarco": {"q_prompt": "Retrieve a web paragraph that answers the following", "d_prompt": "web paragraph"},

    # prompts added by us
    "cqadupstack": {"q_prompt": "Help me to find a related question and its body asked on StackExchange", "d_prompt": "stack exchange question body"},
    "bioasq": {"q_prompt": "Retrieve Scientific paper paragraph to answer this question", "d_prompt": "pubmed abstract"},
    "signal1m": {"q_prompt": "Retrieve tweets relevant to the news article title following", "d_prompt": "tweet"},
    "trec-news": {"q_prompt": "Retrieve relevant news articles that provide background information about the following news headline", "d_prompt": "news article"},
    "robust04": {"q_prompt": "Retrieve relevant news articles to answer the following", "d_prompt": "news article"},

    # instructions used for X2 evaluations (TART paper table 9: https://arxiv.org/pdf/2211.09260.pdf)
    "wikiqa": {"q_prompt": "Retrieve an answer sentence from Wikipedia"},
    "ambigqa": {"q_prompt": "Retrieve a question that is similar to this"},
    "scifact_x2": {"q_prompt": "Retrieve scientific evidence to verify this claim"},
    "gooaq-technical": {"q_prompt": "Find a StackExchange forum that answers this question"},
    "codesearchnet-py": {"q_prompt": "Retrieve a python code that implements the following feature"},
    "linkso-py": {"q_prompt": "You have to find a python implementation of this"},


    # instructions used for training (https://github.com/facebookresearch/tart/blob/main/BERRI/berri_instructions.tsv)
    "nq": {
        "dataset": "nq",
        "q_prompt": "retrieve passages from Wikipedia that provides answers to the following question",
        "d_prompt": "wikipedia paragraph",
        "prompt_1": "retrieve passages from Wikipedia that provides answers to the following question",
        "prompt_2": "You have to find a Wikipedia paragraph that provides the answer to the question",
        "prompt_3": "I want to find an answer for question. Can you find some paragraphs that provide evidence from Wikipedia?",
        "prompt_4": "I'm looking for a Wikipedia paragraph that answers this question. ",
        "prompt_5": "Give me a Wikipedia paragraph answering this open-domain question. ",
        "prompt_6": "A Wikipedia paragraph providing sufficient evidence to answer this question",
        "prompt_7": "Your job is to find a Wikipedia paragraph that answers my question",
        "prompt_8": "You need to retrieve an evidence paragraph from Wikipedia to answer this question"
    },
    "quora": {
        "dataset": "quora",
        "q_prompt": "Find a question that is duplicated with the following question asked in Quora, a community QA forum",
        "d_prompt": "Quora question",
        "prompt_1": "Find a question that is duplicated with the following question asked in Quora, a community QA forum",
        "prompt_2": "I want to find a question similar to this question already asked in Quora. Retrieve a question body that is similar to",
        "prompt_3": "Check if a Quora question is duplicated with this question",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "fever": {
        "dataset": "fever",
        "q_prompt": "Retrieve a Wikipedia paragraph to verify this claim",
        "d_prompt": "wikipedia paragraph",
        "prompt_1": "Retrieve a Wikipedia paragraph to verify this claim",
        "prompt_2": "Find an evidence paragraph from Wikipedia to confirm the statement is correct",
        "prompt_3": "I want to know if this sentence is a fact or not. Can you find relented Wikipedia passages for me?",
        "prompt_4": "You need to find Wikipedia paragraphs that support or refute this sentence",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "hotpotqa": {
        "dataset": "hotpotqa",
        "q_prompt": "Retrieve a Wikipedia paragraph that provides useful evidence or answer for this question",
        "d_prompt": "wikipedia paragraph",
        "prompt_1": "Retrieve a Wikipedia paragraph that provides useful evidence or answer for this question",
        "prompt_2": "I want to know the answer of this question. Please find related Wikipedia passages for me.",
        "prompt_3": "Find a paragraph that provides useful information to answer this question",
        "prompt_4": "You need to find multiple Wikipedia passages to answer this multi-hop question. Can you find related passages?",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "altlex": {
        "dataset": "altlex",
        "q_prompt": "Find an sentence from Simple Wikipedia that corresponds to the following Wikipedia sentence",
        "prompt_1": "Find an sentence from Simple Wikipedia that corresponds to the following Wikipedia sentence",
        "prompt_2": "Retrieve a sentence that talks about the same as the following in a simple or shorter English",
        "prompt_3": "A simplified sentence from Simple Wikipedia of this Wikipedia sentence",
        "prompt_4": "You need to find a simplified sentence from Simple Wikipedia corresponding to the following sentence",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "cnn_dailymail": {
        "dataset": "cnn_dailymail",
        "q_prompt": "The following sentences are the summaries of an news article. Find the source news article.",
        "prompt_1": "The following sentences are the summaries of an news article. Find the source news article.",
        "prompt_2": "Retrieve a news article that is summarized as following",
        "prompt_3": "Find the original news article based on which the following highlights are written",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "coco_captions": {
        "dataset": "coco_captions",
        "q_prompt": "Retrieve a image caption sentence that is written for the same image as the following caption",
        "prompt_1": "Retrieve a image caption sentence that is written for the same image as the following caption",
        "prompt_2": "Find a image caption describing the same image as",
        "prompt_3": "Can you find an image caption talking about the same image as",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "codesearch_go": {
        "dataset": "codesearch_go",
        "q_prompt": "Match the following natural language instruction to Go codes",
        "prompt_1": "Match the following natural language instruction to Go codes",
        "prompt_2": "Retrieve Go implementations achieving the following features ",
        "prompt_3": "Find a Go code implementation on GitHub for the following natural language instruction",
        "prompt_4": "I want to find a Go code implementing the following function from Github. ",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "codesearch_java": {
        "dataset": "codesearch_java",
        "q_prompt": "Match the following natural language instruction to Java codes",
        "prompt_1": "Match the following natural language instruction to Java codes",
        "prompt_2": "Retrieve Java implementations achieving the following features ",
        "prompt_3": "Find a Java code implementation on GitHub for the following natural language instruction",
        "prompt_4": "I want to find Java code implementing the following function from Github. ",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "codesearch_javascript": {
        "dataset": "codesearch_javascript",
        "q_prompt": "Match the following natural language instruction to JavaScript codes",
        "prompt_1": "Match the following natural language instruction to JavaScript codes",
        "prompt_2": "Retrieve JavaScript implementations achieving the following features ",
        "prompt_3": "Find a JavaScript code implementation on GitHub for the following natural language instruction",
        "prompt_4": "I want to find JavaScript code implementing the following function from Github. ",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "codesearch_ruby": {
        "dataset": "codesearch_ruby",
        "q_prompt": "Match the following natural language instruction to Ruby codes",
        "prompt_1": "Match the following natural language instruction to Ruby codes",
        "prompt_2": "Retrieve Ruby implementations achieving the following features ",
        "prompt_3": "Find a Ruby code implementation on GitHub for the following natural language instruction",
        "prompt_4": "I want to find Ruby code implementing the following function from Github. ",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "eli5_question_answer": {
        "dataset": "eli5_question_answer",
        "q_prompt": "Find an answer to this question for me.",
        "prompt_1": "Find an answer to this question for me.",
        "prompt_2": "Retrieve an paragraph-length answer to this question.",
        "prompt_3": "You have to find answer to this question.",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "gigaword": {
        "dataset": "gigaword",
        "q_prompt": "Retrieve a extremely short summary of the following Gigaword article",
        "prompt_1": "Retrieve a extremely short summary of the following Gigaword article",
        "prompt_2": "Find a corresponding headline of the this Gigaword article summary.",
        "prompt_3": "I want to retrieve a headline of this gigaword news",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "yahoo_answers_title_answer": {
        "dataset": "yahoo_answers_title_answer",
        "q_prompt": "Find an answer from a community QA forum for the following question",
        "prompt_1": "Find an answer from a community QA forum for the following question",
        "prompt_2": "Retrieve the most voted answer for this question from Yahoo Answers.",
        "prompt_3": "This is an question's title posted in Yahoo Answers, a community forum. Please retrieve the answer post.",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "mdmcqa": {
        "dataset": "mdmcqa",
        "q_prompt": "Find an evidence from medical text book to answer this medical exam question",
        "prompt_1": "Find an evidence from medical text book to answer this medical exam question",
        "prompt_2": "I want to know the answer of this following medical exam question. Retrieve an evidence passage answering question",
        "prompt_3": "Find the explanation for the correct answer of this medical question",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "medical_sim": {
        "dataset": "medical_sim",
        "q_prompt": "Please retrieve a medical paper summary that is written in a simple language so that my patient can understand",
        "prompt_1": "Please retrieve a medical paper summary that is written in a simple language so that my patient can understand",
        "prompt_2": "You need to find a simple summary of medical paper that corresponds to the following paper abstract",
        "prompt_3": "You need to retrieve a medical paper summary written in a simple English of this paper",
        "prompt_4": "Match this following medical paper abstract to a simplified English summary for non experts",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "msmarco-triplets": {
        "dataset": "msmarco-triplets",
        "q_prompt": "I want to know the answer to the question. Can you find good evidence on the web?",
        "d_prompt": "web paragraph",
        "prompt_1": "I want to know the answer to the question. Can you find good evidence on the web?",
        "prompt_2": "Retrieve a web paragraph that answers the following",
        "prompt_3": "Find an evidence paragraph on the web with evidence and answer for this",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "multilexsum": {
        "dataset": "multilexsum",
        "q_prompt": "Find a short summary of this following legal case",
        "prompt_1": "Find a short summary of this following legal case",
        "prompt_2": "Map this legal case summary to a sentence-long summary",
        "prompt_3": "An extremely short summary sentence of this legal case",
        "prompt_4": "Retrieve a one-sentence summary of the following legal case",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "oqa": {
        "dataset": "oqa",
        "q_prompt": "Find a question that is paraphrased of this",
        "prompt_1": "Find a question that is paraphrased of this",
        "prompt_2": "Retrieve a question that is duplicated with the following",
        "prompt_3": "You need to find duplicate questions in Wiki forum. Could you find a question that is similar to this question",
        "prompt_4": "Find an open-domain question that is similar to the following",
        "prompt_5": "An open-domain question that is duplicated with the following",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "agnews": {
        "dataset": "agnews",
        "q_prompt": "Find a news summary sentence corresponding to the following header",
        "prompt_1": "Find a news summary sentence corresponding to the following header",
        "prompt_2": "Retrieve an sentence-long news summary for this header",
        "prompt_3": "Please find a good summary of this news from the news summary collection",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "pubmedqa": {
        "dataset": "pubmedqa",
        "q_prompt": "Find a related medical paper to answer the following question",
        "prompt_1": "Find a related medical paper to answer the following question",
        "prompt_2": "I want to check if the following statement is true or not. Retrieve a scientific paper from PubMed for me",
        "prompt_3": "Help me to find a highly related pubmed paper to answer this question",
        "prompt_4": "Retrieve a medical paper abstract answering the following question",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "qrecc": {
        "dataset": "qrecc",
        "q_prompt": "Find a meaningful dialogue response to answer the user's question",
        "prompt_1": "Find a meaningful dialogue response to answer the user's question",
        "prompt_2": "Retrieve a good dialogue response that answers this question",
        "prompt_3": "You need to find a good response from a collection of previous responses and help users to know this topic more",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "record": {
        "dataset": "record",
        "q_prompt": "Find a News article to verify the following sentence",
        "prompt_1": "Find a News article to verify the following sentence",
        "prompt_2": "I want to know if this sentence is true. Please retrieve a highly-relevant news article",
        "prompt_3": "News articles that provide a piece of sufficient evidence to verify the following statement",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "scitldr": {
        "dataset": "scitldr",
        "q_prompt": "Find a sentence-length summary of this paper.",
        "prompt_1": "Find a sentence-length summary of this paper.",
        "prompt_2": "Retrieve a really short summary of this paper abstract.",
        "prompt_3": "What is the TLDR of this paper?",
        "prompt_4": "Retrieve a sentence that summarizes the following abstract",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "searchQA_top5_snippets": {
        "dataset": "searchQA_top5_snippets",
        "q_prompt": "Pick up the top web search results' snippets for the following question.",
        "prompt_1": "Pick up the top web search results' snippets for the following question.",
        "prompt_2": "Find the top 5 Web snippets that answer this",
        "prompt_3": "You have to match this question to top five web search snippets.",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "sentence-compression": {
        "dataset": "sentence-compression",
        "q_prompt": "Find a short sentence that compresses the following long sentence",
        "prompt_1": "Find a short sentence that compresses the following long sentence",
        "prompt_2": "You have to match this long sentence to a shorter compressed one",
        "prompt_3": "Retrieve a compressed version of the following sentence written by human annotators",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "npr": {
        "dataset": "npr",
        "q_prompt": "Given a news article headline published at npr.org, find a corresponding summary of the news",
        "prompt_1": "Given a news article headline published at npr.org, find a corresponding summary of the news",
        "prompt_2": "Retrieve a news summary of the following news headline",
        "prompt_3": "You have to match the following news article to a short summary ",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "squad_pairs": {
        "dataset": "squad_pairs",
        "q_prompt": "Find a Wikipedia paragraph that answer the question",
        "prompt_1": "Find a Wikipedia paragraph that answer the question",
        "prompt_2": "You have to retrieve a paragraph from Wikipedia that provides evidence and answer for this question",
        "prompt_3": "This question asks about the details written in a Wikipedia paragraph. Select the paragraph the question is about",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "stackexchange_duplicate_questions_title-body_title-body": {
        "dataset": "stackexchange_duplicate_questions_title-body_title-body",
        "q_prompt": "Find a question paragraph that is duplicated with the following question paragraph at StackExchange.",
        "prompt_1": "Find a question paragraph that is duplicated with the following question paragraph at StackExchange.",
        "prompt_2": "I want to find a question similar to this question already asked in StackExchange. Retrieve a question (main text) that is similar to this following paragraph",
        "prompt_3": "StackExchange is a community QA forum for diverse topics including technical or science. Help me to find a question paragraph that duplicates with my following question paragraph",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "stackexchange_duplicate_questions_title_title": {
        "dataset": "stackexchange_duplicate_questions_title_title",
        "q_prompt": "Find an duplicated question on StackExchange, a community forum.",
        "prompt_1": "Find an duplicated question on StackExchange, a community forum.",
        "prompt_2": "Find a question title that is similar to the following question title asked in StackExchange, a community QA forum",
        "prompt_3": "I want to find a related question asked in StackExchange. Can you find one for me?",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "paq": {
        "dataset": "paq",
        "q_prompt": "Find a web paragraph long answer for this question",
        "prompt_1": "Find a web paragraph long answer for this question",
        "prompt_2": "Can you answer my question by finding an article on the web?",
        "prompt_3": "Retrieve a paragraph answer for the following question",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "triviaqa": {
        "dataset": "triviaqa",
        "q_prompt": "retrieve a related web article that provides evidences and answers for the following Trivia question",
        "prompt_1": "retrieve a related web article that provides evidences and answers for the following Trivia question",
        "prompt_2": "You have to find a answer paragraph on the web for this question",
        "prompt_3": "I want to find an answer for the following Trivia questions. Can you find some paragraphs that provide evidence on the web?",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "wikihow": {
        "dataset": "wikihow",
        "q_prompt": "Find the how-to article to achieve the following goal from Wikihow, a website collecting how-to articles.",
        "prompt_1": "Find the how-to article to achieve the following goal from Wikihow, a website collecting how-to articles.",
        "prompt_2": "WikiHow is an wiki-style community and database of how-to guides. Suggest the best article for the following",
        "prompt_3": "Find a detailed paragraph from WikiHow that explains how-to to achieve",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "wow": {
        "dataset": "wow",
        "q_prompt": "Find an Wikipedia paragraph that is related to the current conversation topic to generate a meaningful response.",
        "prompt_1": "Find an Wikipedia paragraph that is related to the current conversation topic to generate a meaningful response.",
        "prompt_2": "Retrieve a paragraph from Wikipedia to help a conversational AI to generate a knowledge-grounded dialogue",
        "prompt_3": "You must find a Wikipedia paragraph to help a chatbot to generate informative response. Can you find one?",
        "prompt_4": "Find an Wikipedia paragraph related to the following conversation topic.",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "xsum": {
        "dataset": "xsum",
        "q_prompt": "Find a short summary of this following legal case",
        "prompt_1": "Find a short summary of this following legal case",
        "prompt_2": "Map this legal case summary to a sentence-long summary",
        "prompt_3": "An extremely short summary sentence of this legal case",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "ccnews": {
        "dataset": "ccnews",
        "q_prompt": "Retrieve a news article that corresponds to this title",
        "prompt_1": "Retrieve a news article that corresponds to this title",
        "prompt_2": "Find the original news article that this title is written for",
        "prompt_3": "I want to know the details of this news. Can you find a detailed news article on this for me?",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    },
    "wow_response": {
        "dataset": "wow response",
        "q_prompt": "Find a knowledgeable response for an AI chat bot given the following user's inputs",
        "prompt_1": "Find a knowledgeable response for an AI chat bot given the following user's inputs",
        "prompt_2": "Retrieve a informative knowledge-grounded dialogue response given this AI-user chat history",
        "prompt_3": "A good chat bot response to answer this user's query",
        "prompt_4": "",
        "prompt_5": "",
        "prompt_6": "",
        "prompt_7": "",
        "prompt_8": ""
    }

}