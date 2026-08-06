import copy
from src.prompt.base_prompt import AutoPrompt
from typing import Dict


sfr_prompts_map_deprecated = {
    # instructions used for BEIR evaluations (TART paper table 9: https://arxiv.org/pdf/2211.09260.pdf)
    'allnli': {'q_prompt': 'Represent this sentence to retrieve sentences that are semantically entailed by it',
               'd_prompt': 'short sentence'},
    "trec-covid": {
        "q_prompt": "Retrieve PubMed paper to answer this question",
        "d_prompt": "PubMed biomedical article",
        "q_desc": "a question related to the COVID-19 pandemic",
        "d_desc": "Biomedical scientific article from PubMed related to the COVID-19 pandemic"},
    "nfcorpus": {"q_prompt": "Retrieve scientific paper paragraph to answer this question",
                 "d_prompt": "scientific paper paragraph",
                 "q_domain": "question related to health and nutrition",
                 "d_desc": "PubMed article"},
    "fiqa": {"q_prompt": "Find financial web article paragraph to answer",
             "d_prompt": "StackExchange posts about Investment",
             "q_domain": "question related to financial and investment",
             "d_desc": "StackExchange posts under the Investment topic"},
    "arguana": {"q_prompt": "Retrieve an argument that counter argues the following paragraph",
                "d_prompt": "counterargument paragraph",
                "q_desc": "an argument from an online debate portal",
                "d_desc": "a counterargument from an online debate portal"},
    "webis-touche2020": {
        "q_prompt": "You have to retrieve an argument to this debate question",
        "d_prompt": "argument paragraph",
        "q_desc": "an argument from an online debate portal args.me",
        "d_desc": "an argument paragraph from args.me"},
    "dbpedia-entity": {
        "q_prompt": "Retrieve a Wikipedia introduction paragraph of the following entity",
        "d_prompt": "Wikipedia introduction paragraph",
        "q_desc": "a Wikipedia entity",
        "d_desc": "an introduction paragraph about an entity on Wikipedia"},
    "scidocs": {
        "q_prompt": "Find scientific papers that are likely to be cited by the following",
        "d_prompt": "title and abstract of a scientific paper",
        "q_desc": "title of a scientific paper",
        "d_desc": "title and abstract of a scientific paper"},
    "climate-fever": {
        "q_prompt": "I want to know if the following claim is true or not. Retrieve a Wikipedia paragraph on climate change for this.",
        "d_prompt": "Wikipedia article",
        "q_desc": "a claim related to climate change to be verified",
        "d_desc": "an article from Wikipedia which content can be empty"},
    "scifact": {
        "q_prompt": "Retrieve a scientific paper to verify if the following claim is true",
        "d_prompt": "title and abstract of a scientific paper",
        "q_desc": "a scientific claim",
        "d_desc": "title and abstract of a scientific paper"},
    "scifact-title": {
        "q_prompt": "Retrieve a scientific paper to verify if the following claim is true",
        "d_prompt": "abstract of a scientific paper",
        "q_desc": "a scientific claim",
        "d_desc": "abstract of a scientific paper"},

    # other BEIR instructions from training (https://github.com/facebookresearch/tart/blob/main/BERRI/berri_instructions.tsv)
    "msmarco": {
        "q_prompt": "Retrieve a webpage paragraph to answer this question",
        "d_prompt": "webpage paragraph",
        "q_desc": "generic query/question for web search",
        "d_desc": "web page"},
    "fever": {
        "q_prompt": "Retrieve a Wikipedia article to verify the following claim is true or not",
        "d_prompt": "Wikipedia article",
        "q_desc": "a claim to be verified",
        "d_desc": "an article from Wikipedia which content can be empty"},
    "nq": {
        "q_prompt": "Retrieve a Wikipedia paragraph to answer this question",
        "d_prompt": "Wikipedia paragraph",
        "q_desc": "a question can be answered by evidence in Wikipedia",
        "d_desc": "a short paragraph from Wikipedia, can be part of an Wikipedia article"},
    "hotpotqa": {
        "q_prompt": "Retrieve a Wikipedia paragraph to answer this question",
        "d_prompt": "Wikipedia introduction paragraph",
        "q_desc": "a question can be answered by evidence from Wikipedia",
        "d_desc": "an introduction paragraph about an entity on Wikipedia"},
    "quora": {
        "q_prompt": "Check if a Quora question is duplicated with this question.",
        "d_prompt": "Quora question",
        "q_desc": "a question/title asked on Quora, which can be duplicate",
        "d_desc": "a question/title asked on Quora, which can be duplicate"},
    "cqadupstack": {
        "q_prompt": "Help me to find a related question and its body asked on StackExchange",
        "d_prompt": "StackExchange question and body",
        "q_desc": "a question asked on StackExchange",
        "d_desc": "a question and its body asked on StackExchange"},
    "bioasq": {"q_prompt": "Retrieve bio-medical scientific paper paragraph to answer this question",
               "d_prompt": "title and abstract of a PubMed paper",
               "q_desc": "a question about bio-medical science",
               "d_desc": "title and abstract of a PubMed paper"},
    "signal1m": {"q_prompt": "Retrieve tweets relevant to the news article title following",
                 "d_prompt": "tweet related to news discussions",
                 "q_desc": "a news topic that causes discussions on twitter",
                 "d_desc": "tweet related to news discussions"},
    "trec-news": {"q_prompt": "Retrieve relevant news articles that provide background information about the following news headline",
                  "d_prompt": "Washington Post news article",
                  "q_desc": "a news headline",
                  "d_desc": "Washington Post news article"},
    "robust04": {"q_prompt": "Retrieve relevant news articles to answer the following",
                 "d_prompt": "news article",
                 "q_desc": "a query related to news",
                 "d_desc": "news article"},

    # instructions used in BERRI (TART training) (https://github.com/facebookresearch/tart/blob/main/BERRI/berri_instructions.tsv)
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
    "fever_berri": {
        "dataset": "fever",
        "q_prompt": "Retrieve a Wikipedia paragraph to verify this claim",
        "prompt_1": "Retrieve a Wikipedia paragraph to verify this claim",
        "prompt_2": "Find an evidence paragraph from Wikipedia to confirm the statement is correct",
        "prompt_3": "I want to know if this sentence is a fact or not. Can you find relented Wikipedia passages for me?",
        "prompt_4": "You need to find Wikipedia paragraphs that support or refute this sentence",
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
    "hotpotqa_berri": {
        "dataset": "hotpotqa",
        "q_prompt": "Retrieve a Wikipedia paragraph that provides useful evidence or answer for this question",
        "prompt_1": "Retrieve a Wikipedia paragraph that provides useful evidence or answer for this question",
        "prompt_2": "I want to know the answer of this question. Please find related Wikipedia passages for me.",
        "prompt_3": "Find a paragraph that provides useful information to answer this question",
        "prompt_4": "You need to find multiple Wikipedia passages to answer this multi-hop question. Can you find related passages?",
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
    "nq_berri": {
        "dataset": "nq",
        "q_prompt": "retrieve passages from Wikipedia that provides answers to the following question",
        "prompt_1": "retrieve passages from Wikipedia that provides answers to the following question",
        "prompt_2": "You have to find a Wikipedia paragraph that provides the answer to the question",
        "prompt_3": "I want to find an answer for question. Can you find some paragraphs that provide evidence from Wikipedia?",
        "prompt_4": "I'm looking for a Wikipedia paragraph that answers this question. ",
        "prompt_5": "Give me a Wikipedia paragraph answering this open-domain question. ",
        "prompt_6": "A Wikipedia paragraph providing sufficient evidence to answer this question",
        "prompt_7": "Your job is to find a Wikipedia paragraph that answers my question",
        "prompt_8": "You need to retrieve an evidence paragraph from Wikipedia to answer this question"
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
    "quora_berri": {
        "dataset": "quora",
        "q_prompt": "Find a question that is duplicated with the following question asked in Quora, a community QA forum",
        "prompt_1": "Find a question that is duplicated with the following question asked in Quora, a community QA forum",
        "prompt_2": "I want to find a question similar to this question already asked in Quora. Retrieve a question body that is similar to",
        "prompt_3": "Check if a Quora question is duplicated with this question",
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

MTOP_INTENT_LABELS = ['add_time_timer', 'add_to_playlist_music', 'answer_call', 'cancel_call',
                    'cancel_message', 'create_alarm', 'create_call', 'create_playlist_music',
                    'create_reminder', 'create_timer', 'delete_alarm', 'delete_playlist_music', 'delete_reminder',
                    'delete_timer', 'dislike_music', 'disprefer', 'end_call', 'fast_forward_music',
                    'follow_music', 'get_age', 'get_airquality', 'get_alarm', 'get_attendee_event',
                    'get_availability', 'get_call', 'get_call_contact', 'get_call_time', 'get_category_event',
                    'get_contact', 'get_contact_method', 'get_date_time_event', 'get_details_news',
                    'get_education_degree', 'get_education_time', 'get_employer', 'get_employment_time',
                    'get_event', 'get_gender', 'get_group', 'get_info_contact', 'get_info_recipes', 'get_job',
                    'get_language', 'get_life_event', 'get_life_event_time', 'get_location', 'get_lyrics_music',
                    'get_major', 'get_message', 'get_message_contact', 'get_mutual_friends', 'get_recipes',
                    'get_reminder', 'get_reminder_amount', 'get_reminder_date_time', 'get_reminder_location',
                    'get_stories_news', 'get_sunrise', 'get_sunset', 'get_timer', 'get_track_info_music',
                    'get_undergrad', 'get_weather', 'help_reminder', 'hold_call', 'ignore_call', 'is_true_recipes',
                    'like_music', 'loop_music', 'merge_call', 'pause_music', 'pause_timer', 'play_media',
                    'play_music', 'prefer', 'previous_track_music', 'question_music', 'question_news',
                    'remove_from_playlist_music', 'repeat_all_music', 'repeat_all_off_music', 'replay_music',
                    'restart_timer', 'resume_call', 'resume_music', 'resume_timer', 'rewind_music', 'send_message',
                    'set_available', 'set_default_provider_calling', 'set_default_provider_music',
                    'set_rsvp_interested', 'set_rsvp_no', 'set_rsvp_yes', 'set_unavailable', 'share_event',
                    'silence_alarm', 'skip_track_music', 'snooze_alarm', 'start_shuffle_music', 'stop_music',
                    'stop_shuffle_music', 'subtract_time_timer', 'switch_call', 'unloop_music', 'update_alarm',
                    'update_call', 'update_method_call', 'update_reminder', 'update_reminder_date_time',
                    'update_reminder_location', 'update_reminder_todo', 'update_timer']


CLUSTERING_NAME2LABELS = {
    'arxiv': ['acc-phys', 'adap-org', 'alg-geom', 'ao-sci', 'astro-ph', 'astro-ph.CO', 'astro-ph.EP', 'astro-ph.GA', 'astro-ph.HE', 'astro-ph.IM', 'astro-ph.SR', 'atom-ph', 'chao-dyn', 'chem-ph', 'cmp-lg', 'comp-gas', 'cond-mat', 'cond-mat.dis-nn', 'cond-mat.mes-hall', 'cond-mat.mtrl-sci', 'cond-mat.other', 'cond-mat.quant-gas', 'cond-mat.soft', 'cond-mat.stat-mech', 'cond-mat.str-el', 'cond-mat.supr-con', 'cs', 'cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL', 'cs.CR', 'cs.CV', 'cs.CY', 'cs.DB', 'cs.DC', 'cs.DL', 'cs.DM', 'cs.DS', 'cs.ET', 'cs.FL', 'cs.GL', 'cs.GR', 'cs.GT', 'cs.HC', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO', 'cs.MA', 'cs.MM', 'cs.MS', 'cs.NA', 'cs.NE', 'cs.NI', 'cs.OH', 'cs.OS', 'cs.PF', 'cs.PL', 'cs.RO', 'cs.SC', 'cs.SD', 'cs.SE', 'cs.SI', 'cs.SY', 'dg-ga', 'econ', 'econ.EM', 'econ.GN', 'econ.TH', 'eess', 'eess.AS', 'eess.IV', 'eess.SP', 'eess.SY', 'funct-an', 'gr-qc', 'hep', 'hep-ex', 'hep-lat', 'hep-ph', 'hep-th', 'math', 'math-ph', 'math.AC', 'math.AG', 'math.AP', 'math.AT', 'math.CA', 'math.CO', 'math.CT', 'math.CV', 'math.DG', 'math.DS', 'math.FA', 'math.GM', 'math.GN', 'math.GR', 'math.GT', 'math.HO', 'math.KT', 'math.LO', 'math.MG', 'math.NA', 'math.NT', 'math.OA', 'math.OC', 'math.PR', 'math.QA', 'math.RA', 'math.RT', 'math.SG', 'math.SP', 'math.ST', 'mtrl-th', 'nlin.AO', 'nlin.CD', 'nlin.CG', 'nlin.PS', 'nlin.SI', 'nucl-ex', 'nucl-th', 'other', 'patt-sol', 'physics.acc-ph', 'physics.ao-ph', 'physics.app-ph', 'physics.atm-clus', 'physics.atom-ph', 'physics.bio-ph', 'physics.chem-ph', 'physics.class-ph', 'physics.comp-ph', 'physics.data-an', 'physics.ed-ph', 'physics.flu-dyn', 'physics.gen-ph', 'physics.geo-ph', 'physics.hist-ph', 'physics.ins-det', 'physics.med-ph', 'physics.optics', 'physics.plasm-ph', 'physics.pop-ph', 'physics.soc-ph', 'physics.space-ph', 'plasm-ph', 'q-alg', 'q-bio', 'q-bio.BM', 'q-bio.CB', 'q-bio.GN', 'q-bio.MN', 'q-bio.NC', 'q-bio.OT', 'q-bio.PE', 'q-bio.QM', 'q-bio.SC', 'q-bio.TO', 'q-fin', 'q-fin.CP', 'q-fin.EC', 'q-fin.GN', 'q-fin.MF', 'q-fin.PM', 'q-fin.PR', 'q-fin.RM', 'q-fin.ST', 'q-fin.TR', 'quant-ph', 'solv-int', 'stat', 'stat.AP', 'stat.CO', 'stat.ME', 'stat.ML', 'stat.OT', 'supr-con'],
    'biorxiv': ['animal behavior and cognition', 'biochemistry', 'bioengineering', 'bioinformatics', 'biophysics', 'cancer biology', 'cell biology', 'developmental biology', 'ecology', 'epidemiology', 'evolutionary biology', 'genetics', 'genomics', 'immunology', 'microbiology', 'molecular biology', 'neuroscience', 'paleontology', 'pathology', 'pharmacology and toxicology', 'physiology', 'plant biology', 'scientific communication and education', 'synthetic biology', 'systems biology', 'zoology'],
    'medrxiv': ['addiction medicine', 'allergy and immunology', 'anesthesia', 'cardiovascular medicine', 'dentistry and oral medicine', 'dermatology', 'emergency medicine', 'endocrinology', 'epidemiology', 'forensic medicine', 'gastroenterology', 'genetic and genomic medicine', 'geriatric medicine', 'health economics', 'health informatics', 'health policy', 'health systems and quality improvement', 'hematology', 'hiv aids', 'infectious diseases', 'intensive care and critical care medicine', 'medical education', 'medical ethics', 'nephrology', 'neurology', 'nursing', 'nutrition', 'obstetrics and gynecology', 'occupational and environmental health', 'oncology', 'ophthalmology', 'orthopedics', 'otolaryngology', 'pain medicine', 'palliative medicine', 'pathology', 'pediatrics', 'pharmacology and therapeutics', 'primary care research', 'psychiatry and clinical psychology', 'public and global health', 'radiology and imaging', 'rehabilitation medicine and physical therapy', 'respiratory medicine', 'rheumatology', 'sexual and reproductive health', 'sports medicine', 'surgery', 'toxicology', 'transplantation', 'urology'],
    'reddit': ['1200isplentyketo', '196', '40kLore', '90dayfianceuncensored', 'ABA', 'ABraThatFits', 'ACTrade', 'ADHD', 'AFL.txt', 'Adopted', 'Advice.txt', 'Aerials', 'AgonGame', 'AirForce', 'AirReps', 'Aliexpress', 'AmItheAsshole', 'AmazonFC', 'AmongUs', 'Anarchism.txt', 'Animals.txt', 'ApplyingToCollege', 'Art.txt', 'AskAnAmerican', 'AskCulinary', 'AskGames', 'AskGirls', 'AskLosAngeles', 'AskReddit.txt', 'AskVet', 'AssassinsCreedValhala', 'AstroGaming', 'Astronomy.txt', 'Atlanta.txt', 'Austin.txt', 'Autos.txt', 'BDSMcommunity', 'BMW', 'BangaloreMains', 'Bariloche', 'Bath', 'Baystreetbets', 'BenignExistence', 'Bestbuy', 'Bible', 'BitLifeApp', 'BlackCountryNewRoad', 'Blogging.txt', 'Bogleheads', 'BokunoheroFanfiction', 'Boxing.txt', 'BreakUps', 'BroadCity', 'Buddhism.txt', 'BuddyCrossing', 'C25K', 'COVID19positive', 'CPTSD', 'CRISPR', 'CanaryWharfBets', 'Cartalk', 'Catholicism', 'Catholicism.txt', 'CharacterRant', 'China.txt', 'Christianity.txt', 'ChronicPain', 'ClashOfClans', 'ClashOfClansRecruit', 'Coffee.txt', 'Coins4Sale', 'ConnecticutR4R', 'ConquerorsBlade', 'CozyGrove', 'CreditCards', 'CryptoHorde', 'CryptoMarkets', 'CryptoMars', 'Crypto_com', 'CultoftheFranklin', 'DMAcademy', 'DestinyTheGame', 'DissonautUniverse', 'DnD', 'Dodocodes', 'Doom', 'DotA2', 'Dyson_Sphere_Program', 'ELLIPAL_Official', 'Eldenring', 'EliteDangerous', 'EscapefromTarkov', 'EverMerge', 'Evernote', 'FIFA', 'FanFiction', 'FiddlesticksMains', 'FiestaST', 'FireEmblemHeroes', 'FitnessDE', 'FortNiteBR', 'FortniteCompetitive', 'FridayNightFunkin', 'GCXRep', 'GME', 'GameBuilderGarage', 'GameSale', 'GamingLaptops', 'Gaming_Headsets', 'GayYoungOldDating', 'GearsOfWar', 'GlobalPowers', 'GoRVing', 'GodEater', 'GoodNotes', 'GoogleDataStudio', 'GriefSupport', 'Gta5Modding', 'Guitar', 'GunAccessoriesForSale', 'HFY', 'HIMYM', 'HaircareScience', 'HauntingOfHillHouse', 'Hedgehog', 'HilariaBaldwin', 'HiveOS', 'HotWheels', 'Huel', 'HyperRP', 'HypnoFair', 'IBO', 'IRS', 'ITCareerQuestions', 'InstacartShoppers', 'JamesHoffmann', 'JoJolion', 'Julia', 'JustUnsubbed', 'KGBTR', 'KTM', 'KassadinMains', 'Kayaking', 'KazuhaMains', 'Kengan_Ashura', 'KikRoleplay', 'KillingEve', 'KingkillerChronicle', 'LOONA', 'LeagueConnect', 'LearnerDriverUK', 'LegendsOfRuneterra', 'Library', 'LifeProTips', 'LightNovels', 'MECoOp', 'MLBTheShow', 'MachineLearning', 'Market76', 'MarsWallStreet', 'MarylandUnemployment', 'MatthiasSubmissions', 'McMaster', 'MensRights', 'Midsommar', 'MindHunter', 'Minecraft', 'MinecraftServer', 'ModelCars', 'ModernMagic', 'MondoGore', 'MonsterHunter', 'MonsterHunterWorld', 'MtF', 'NBA2k', 'NTU', 'NameThatSong', 'NassauCountyHookups', 'Netherlands', 'NiceHash', 'Nioh', 'NoContract', 'NoFap', 'NoFeeAC', 'NoMansSkyTheGame', 'OculusQuest', 'Paladins', 'PanicAttack', 'PerkByDaylight', 'PersonalFinanceCanada', 'PokemonGoFriends', 'PokemonHome', 'PokemonTCG', 'PoonamPandeyFanatics', 'Prolactinoma', 'Random_Acts_Of_Amazon', 'RedditWritesSeinfeld', 'Reduction', 'RepTime', 'RevenantMain', 'Revu', 'RocketLeagueExchange', 'RoleplayingForReddit', 'RomanianWolves', 'SLFmeetups', 'STAYC', 'SakuraGakuin', 'SaltLakeCity', 'Sat', 'SatoshiStreetBets', 'SauceSharingCommunity', 'Scotch', 'Screenwriting', 'ShieldAndroidTV', 'Shittyaskflying', 'ShuumatsuNoValkyrie', 'Sims4', 'SluttyConfessions', 'Smallville', 'SofiawithanF', 'SquaredCircle', 'Stormlight_Archive', 'SummonSign', 'Superhero_Ideas', 'Superstonk', 'Sverigesforsvarsmakt', 'TMJ', 'TeensMeetTeens', 'Testosterone', 'TheMagnusArchives', 'TherosDMs', 'Throwers', 'Tomorrowland', 'TooAfraidToAsk', 'TwoXChromosomes', 'UKJobs', 'Unemployment', 'ValorantBrasil', 'Vent', 'VinylCollectors', 'VirtualYoutubers', 'WallStreetbetsELITE', 'Wallstreetsilver', 'Wavyhair', 'WetlanderHumor', 'WhiteWolfRPG', 'Wordpress', 'WreckingBallMains', 'ZombsRoyale', 'accesscontrol', 'advertising.txt', 'aggretsuko', 'airsoft.txt', 'alcoholicsanonymous', 'aliens', 'amateurradio.txt', 'amazon.txt', 'amcstock', 'amex', 'anime', 'anime.txt', 'aorus', 'apexlegends', 'apple.txt', 'appletv', 'appliancerepair', 'architecture.txt', 'arlington', 'aromantic', 'ask', 'askTO', 'askcarsales', 'askseddit', 'asktransgender', 'astrology.txt', 'atheism.txt', 'auntienetwork', 'australia.txt', 'aviation.txt', 'baltimore.txt', 'beer.txt', 'belgium.txt', 'bicycling', 'bicycling.txt', 'biology.txt', 'blackopscoldwar', 'bleach', 'blender.txt', 'boardgames', 'bonnaroo', 'books.txt', 'boston.txt', 'bostonr4r', 'brasil', 'brasil.txt', 'brave_browser', 'bravelydefault', 'breakingmom', 'buildapc', 'buildapcforme', 'canada.txt', 'cancer.txt', 'candlemagick', 'capitalism_in_decay', 'cardano', 'cars.txt', 'cats.txt', 'chemistry.txt', 'chess.txt', 'chicago.txt', 'choiceofgames', 'chronotrigger', 'classicalmusic.txt', 'clonewars', 'coins.txt', 'collapse.txt', 'college.txt', 'comedy.txt', 'comicbooks', 'comicbooks.txt', 'comicswap', 'conspiracy', 'controlgame', 'covidlonghaulers', 'cryptocoins', 'cryptostreetbets', 'cscareerquestionsEU', 'cuboulder', 'dating', 'dating_advice', 'dbrand', 'deadbydaylight', 'delhi', 'depressed', 'depression', 'destiny2', 'detrans', 'digimon', 'dirtykikpals', 'dirtypenpals', 'dogecoin', 'doordash_drivers', 'dpdr', 'dragonquest', 'dreamsmp', 'emacs', 'emotestories', 'endocrinology', 'eu4', 'exalted', 'exjw', 'exmuslim', 'extrarfl', 'facebook', 'fantasywriters', 'feedthebeast', 'feminineboys', 'findaleague', 'fleshlight', 'foodscience', 'gamegrumps', 'gaming', 'gardening', 'generationology', 'gonewildaudio', 'gorillaz', 'graylog', 'haskell', 'help', 'heroesofthestorm', 'hisdarkmaterials', 'hitbtc', 'hoi4', 'ibs', 'idlechampions', 'idleon', 'igcse', 'imaginarypenpals', 'islam', 'kansascity', 'kdramarecommends', 'kingofqueens', 'ladybusiness', 'langrisser', 'lasers', 'lawschooladmissions', 'leafs', 'leagueoflegends', 'learnmath', 'lfg', 'libraryofruina', 'livesound', 'lockpicking', 'logh', 'mac', 'madisonwi', 'masseffect', 'mechmarket', 'metalgearsolid', 'mfdoom', 'miraculousladybug', 'namenerds', 'nanocurrency', 'newtothenavy', 'nhl', 'nus', 'obs', 'oculus', 'offmychest', 'oneui', 'onewheel', 'overlord', 'paintdotnet', 'passive_income', 'photography', 'piano', 'pittsburgh', 'playstation', 'podcasts', 'pokemon', 'popperpigs', 'premed', 'ps3homebrew', 'puppy101', 'qBittorrent', 'quittingkratom', 'raisedbynarcissists', 'raleigh', 'realonlyfansreviews', 'redsox', 'relationship_advice', 'religion', 'roblox', 'royalfamily', 'rpg', 'salesforce', 'samsung', 'scatstories', 'screenplaychallenge', 'seduction', 'self', 'selfimprovement', 'seoul', 'sex', 'sexstories', 'silverbugbets', 'sissypersonals', 'skincareexchange', 'snuffrp', 'socialanxiety', 'sofi', 'software', 'spirituality', 'stalker', 'starcraft2coop', 'stashinvest', 'steelseries', 'steinsgate', 'stocks', 'suggestmeabook', 'survivinginfidelity', 'tarot', 'techsupport', 'teenagers', 'teenagersnew', 'texas', 'tf2trade', 'thedivision', 'theta_network', 'thetagang', 'thomasthetankengine', 'tipofmypenis', 'tipofmytongue', 'tomorrow', 'touhou', 'trakstocks', 'transgendercirclejerk', 'transpositive', 'turo', 'turtles', 'tutanota', 'u_mawadom118', 'ugly', 'unpopularopinion', 'vaginismus', 'vcu', 'visualization', 'watercooling', 'weather', 'whatcarshouldIbuy', 'whatsthatbook', 'whowouldwin', 'wicked_edge', 'wireshark', 'xbox', 'xboxfindfriends', 'xmen', 'xmpp', 'yoga', 'youtubers'],
    'stackexchange': ['.net', '2.5d', '2d', '3d', '3d-meshes', '3d-modeling', '3dsmax', 'aabb', 'accessibility', 'actionscript', 'actionscript-3', 'adventure-game-studio', 'adventure-games', 'advertisements', 'ai', 'ajax', 'algorithm', 'allegro', 'alpha', 'alpha-blending', 'analytics', 'andengine', 'android', 'android-studio', 'angelscript', 'angles', 'animation', 'anti-cheat', 'antialiasing', 'appstore', 'architecture', 'arma3', 'art', 'artemis', 'aspect-ratio', 'assembly', 'asset-management', 'assets', 'assimp', 'audio', 'augmented-reality', 'authentication', 'automation', 'avatar', 'awesomium', 'babylonjs', 'balance', 'barycentric-coordinates', 'behavior', 'behavior-tree', 'beta', 'beziers', 'blender', 'blender-game-engine', 'board-game', 'books', 'bot', 'bounding-boxes', 'bounding-spheres', 'box2d', 'browser', 'browser-based-games', 'bsp', 'bukkit', 'bullet-physics', 'business', 'business-model', 'c', 'c#', 'c++', 'camera', 'car', 'card-game', 'career', 'casual-games', 'character', 'chess', 'client-server', 'clipping', 'cloud-computing', 'cocos2d', 'cocos2d-iphone', 'cocos2d-x', 'cocos2d-x-js', 'code-reflection', 'collada', 'collider', 'collision-detection', 'collision-resolution', 'color', 'combat', 'commodore-64', 'community-management', 'compatibility', 'competition', 'component-based', 'compression', 'computational-geometry', 'compute-shader', 'configuration', 'console', 'construct-2', 'content-generation', 'content-rating', 'control', 'controllers', 'coordinates', 'copyright', 'corona-sdk', 'costs', 'cross-platform', 'crusader-kings-2-modding', 'cryengine', 'css', 'cubemap', 'culling', 'curves', 'data', 'data-driven', 'data-structure', 'databases', 'debugging', 'deferred-rendering', 'demo', 'deployment', 'depth-buffer', 'design-patterns', 'destructables', 'dev-groups', 'development-speed', 'difficulty', 'direct3d12', 'directinput', 'directx', 'directx10', 'directx11', 'directx9', 'distribution', 'documentation', 'double-buffering', 'dynamic-difficulty', 'eclipse', 'economy', 'editors', 'education', 'effect', 'effects', 'efficiency', 'emulation', 'encryption', 'engineering.stackexchange.com.txt', 'english.stackexchange.com.txt', 'entity', 'entity-component', 'entity-component-system', 'entity-system', 'eosio.stackexchange.com.txt', 'es.stackoverflow.com.txt', 'esperanto.stackexchange.com.txt', 'ethereum.stackexchange.com.txt', 'events', 'expatriates.stackexchange.com.txt', 'expressionengine.stackexchange.com.txt', 'extrapolation', 'face', 'facebook', 'farseer-physics-engine', 'fbx', 'file', 'file-format', 'filesystem', 'first-person-shooter', 'fitness.stackexchange.com.txt', 'fixed-timestep', 'flash', 'flash-develop', 'flixel', 'floating-point', 'fluid-dynamics', 'flutter', 'fmod', 'fonts', 'fractal', 'fragment-shader', 'frame-buffer', 'frame-rate', 'free-to-play', 'freelancing.stackexchange.com.txt', 'french.stackexchange.com.txt', 'frustum', 'frustum-culling', 'fsm', 'functional', 'fund-raising', 'game-center', 'game-design', 'game-industry', 'game-loop', 'game-maker', 'game-maker-dnd', 'game-maker-studio-2', 'game-mechanics', 'game-recording', 'game-state', 'gamedev.stackexchange.com.txt', 'gameobject', 'gamepad', 'gamesalad', 'gaming.stackexchange.com.txt', 'gardening.stackexchange.com.txt', 'gdevelop', 'genealogy.stackexchange.com.txt', 'geolocation', 'geometry', 'german.stackexchange.com.txt', 'gimp', 'gis.stackexchange.com.txt', 'glfw', 'glm', 'global-illumination', 'glsl', 'glut', 'go', 'godot', 'google-app-engine', 'google-play', 'google-play-services', 'gpu', 'graph', 'graphic-effects', 'graphicdesign.stackexchange.com.txt', 'graphics', 'graphics-design', 'graphics-programming', 'grid', 'gui', 'hacks', 'ham.stackexchange.com.txt', 'hammer', 'hardware', 'hardware-acceleration', 'hardwarerecs.stackexchange.com.txt', 'harlowe', 'havok', 'haxe', 'hdr', 'health.stackexchange.com.txt', 'heightmap', 'hermeneutics.stackexchange.com.txt', 'heuristics', 'hexagonal-grid', 'hinduism.stackexchange.com.txt', 'hiring', 'history.stackexchange.com.txt', 'hlsl', 'homebrew', 'homebrew.stackexchange.com.txt', 'hosting', 'hsm.stackexchange.com.txt', 'htc-vive', 'html', 'html-canvas', 'html5', 'hud', 'human-resources', 'ide', 'image', 'image-processing', 'impactjs', 'in-app-purchase', 'index-buffer', 'input', 'installer', 'instancing', 'intellectual-property', 'interactive-fiction', 'interface', 'interpersonal.stackexchange.com.txt', 'interpolation', 'intersection', 'inventory', 'inverse-kinematics', 'ios', 'iot.stackexchange.com.txt', 'iota.stackexchange.com.txt', 'ipad', 'iphone', 'irrlicht', 'islam.stackexchange.com.txt', 'isometric', 'italian.stackexchange.com.txt', 'ja.stackoverflow.com.txt', 'japanese.stackexchange.com.txt', 'java', 'javafx', 'javascript', 'jbox2d', 'jmonkeyengine', 'jobs', 'jogl', 'joomla.stackexchange.com.txt', 'joystick', 'jquery', 'judaism.stackexchange.com.txt', 'jumping', 'keyboard', 'kinect', 'korean.stackexchange.com.txt', 'languagelearning.stackexchange.com.txt', 'latin.stackexchange.com.txt', 'law.stackexchange.com.txt', 'leaderboards', 'legal', 'level-design', 'level-of-detail', 'levels', 'libgdx', 'licensing', 'lifehacks.stackexchange.com.txt', 'lighting', 'line-of-sight', 'linear-algebra', 'linguistics.stackexchange.com.txt', 'linux', 'literature.stackexchange.com.txt', 'loading', 'localization', 'logic', 'love2d', 'lua', 'lumberyard-engine', 'lwjgl', 'macos', 'magento.stackexchange.com.txt', 'magicavoxel', 'management', 'manuals', 'map-editor', 'maps', 'marching-cubes', 'marketing', 'marmalade', 'martialarts.stackexchange.com.txt', 'matchmaking', 'materials', 'math.stackexchange.com.txt', 'matheducators.stackexchange.com.txt', 'mathematica.stackexchange.com.txt', 'mathematics', 'matrix', 'maya', 'maze', 'md5mesh', 'mechanics.stackexchange.com.txt', 'memory', 'memory-efficiency', 'mesh', 'methodology', 'metrics', 'microsoft', 'microtransactions', 'minecraft-modding', 'mission-design', 'mmo', 'mobile', 'modding', 'modelling-techniques', 'models', 'moderators.stackexchange.com.txt', 'monero.stackexchange.com.txt', 'monetization', 'money.stackexchange.com.txt', 'monogame', 'monotouch', 'motion', 'motivation', 'mouse', 'movement', 'movies.stackexchange.com.txt', 'mud', 'multiplayer', 'multithreading', 'music', 'music.stackexchange.com.txt', 'musicfans.stackexchange.com.txt', 'mvc', 'mysql', 'mythology.stackexchange.com.txt', 'navmesh', 'ndk', 'networkengineering.stackexchange.com.txt', 'networking', 'nintendo', 'node.js', 'noise', 'non-photorealistic', 'normal-generation', 'normal-mapping', 'normals', 'npc', 'nvidia', 'obj', 'objective-c', 'objects', 'occlusion', 'octree', 'oculus', 'ogre', 'online', 'oop', 'open-source', 'openal', 'opendata.stackexchange.com.txt', 'opengl', 'opengl-es', 'opengl-es2', 'openscenegraph', 'opensource.stackexchange.com.txt', 'opentk', 'operating-system', 'optimization', 'or.stackexchange.com.txt', 'orientation', 'outdoors.stackexchange.com.txt', 'outsourcing', 'palette', 'panda3d', 'parallax-scrolling', 'parenting.stackexchange.com.txt', 'particles', 'patching', 'patents.stackexchange.com.txt', 'path', 'path-finding', 'pc', 'performance', 'performance-budgeting', 'perlin-noise', 'perspective', 'pets.stackexchange.com.txt', 'phaser', 'philosophy.stackexchange.com.txt', 'photo.stackexchange.com.txt', 'php', 'physics', 'physics-engine', 'physics.stackexchange.com.txt', 'physx', 'pico-8', 'pipeline', 'piracy', 'pixel', 'pixel-art', 'planning', 'platform', 'platformer', 'playn', 'playstation3', 'playstation4', 'playtesting', 'plugin', 'pm.stackexchange.com.txt', 'point-cloud', 'poker.stackexchange.com.txt', 'politics.stackexchange.com.txt', 'polymorphism', 'portals', 'porting', 'portuguese.stackexchange.com.txt', 'post-processing', 'pre-production', 'probability', 'procedural', 'procedural-generation', 'process', 'productivity', 'profiling', 'project-management', 'projectile-physics', 'projection', 'prototyping', 'psm', 'pt.stackoverflow.com.txt', 'publishing', 'puzzle', 'puzzling.stackexchange.com.txt', 'pygame', 'pyglet', 'python', 'quadtree', 'quake2', 'quake3', 'quant.stackexchange.com.txt', 'quantumcomputing.stackexchange.com.txt', 'quaternion', 'racing', 'random', 'ranking', 'raspberrypi.stackexchange.com.txt', 'raycasting', 'raytracing', 'real-money-transactions', 'refactoring', 'release', 'rendering', 'renpy', 'replays', 'resolution', 'resource-management', 'retrocomputing.stackexchange.com.txt', 'revenue', 'reverseengineering.stackexchange.com.txt', 'rigging', 'rigidbody', 'roblox', 'robotics.stackexchange.com.txt', 'rock-paper-scissors', 'roguelikes', 'rope-physics', 'rotation', 'rpg', 'rpg-maker', 'rpg-maker-mv', 'rpg-maker-vx', 'rpg-maker-xp', 'rpg.stackexchange.com.txt', 'rts', 'ru.stackoverflow.com.txt', 'ruby', 'rus.stackexchange.com.txt', 'russian.stackexchange.com.txt', 'sales', 'salesforce.stackexchange.com.txt', 'savegame', 'scala', 'scale', 'scene', 'scene-graph', 'scicomp.stackexchange.com.txt', 'scifi.stackexchange.com.txt', 'scoring', 'screen', 'scripting', 'sdl', 'sdl2', 'security', 'security.stackexchange.com.txt', 'selection', 'separating-axis-theorem', 'serialization', 'server', 'serverfault.com.txt', 'settings', 'sfml', 'shaders', 'shading', 'shadow-mapping', 'shadows', 'sharepoint.stackexchange.com.txt', 'sharpdx', 'side-scroller', 'signed-distance-field', 'silverlight', 'simulations', 'single-player', 'sitecore.stackexchange.com.txt', 'skeletal-animation', 'skeptics.stackexchange.com.txt', 'sketchup', 'sky', 'skybox', 'skyrim-modding', 'slick', 'slimdx', 'smartfox', 'social', 'software-engineering', 'software-rendering', 'softwareengineering.stackexchange.com.txt', 'softwarerecs.stackexchange.com.txt', 'sound', 'sound-effects', 'sound.stackexchange.com.txt', 'source-code', 'source-engine', 'soya3d', 'space-partitioning', 'space.stackexchange.com.txt', 'spaces', 'spanish.stackexchange.com.txt', 'spatial-partitioning', 'spawning', 'special-effects', 'spherical-harmonics', 'splash-screen', 'splines', 'sponza', 'sports.stackexchange.com.txt', 'spritebatch', 'spritekit', 'sprites', 'spritesheet', 'sqa.stackexchange.com.txt', 'sql', 'stackapps.com.txt', 'stackoverflow.com.txt', 'stage3d', 'starcraft-2', 'state', 'statistics', 'stats.stackexchange.com.txt', 'steam', 'steering-behaviors', 'stellar.stackexchange.com.txt', 'stencyl', 'storage', 'storyboard', 'strategy', 'streaming', 'superuser.com.txt', 'sustainability.stackexchange.com.txt', 'swift', 'synchronization', 'teaching', 'teamwork', 'terminology', 'terrain', 'terrain-rendering', 'tessellation', 'testing', 'tetris', 'tex.stackexchange.com.txt', 'text', 'text-based', 'texture-atlas', 'textures', 'tezos.stackexchange.com.txt', 'three.js', 'tiled', 'tilemap', 'tiles', 'time-travel', 'timer', 'timestep', 'timing', 'tools', 'tor.stackexchange.com.txt', 'torque', 'torque-x', 'touch', 'tournament', 'tower-defense', 'trademark', 'trailer', 'trajectory', 'transformation', 'transparency', 'travel.stackexchange.com.txt', 'triangulation', 'tridion.stackexchange.com.txt', 'trigonometry', 'turn-based', 'turn-based-strategy', 'tutorials', 'twine', 'udk', 'udp', 'ui-design', 'ukrainian.stackexchange.com.txt', 'unit-testing', 'unity', 'unity-ads', 'unity-networking', 'unity-webgl', 'unityscript', 'unix.stackexchange.com.txt', 'unreal', 'unreal-4', 'user-experience', 'user-generated-content', 'user-interface', 'uv-mapping', 'ux.stackexchange.com.txt', 'valve', 'vb.net', 'vbo', 'vector', 'vector-art', 'vegetarianism.stackexchange.com.txt', 'version-control', 'vertex', 'vertex-buffer', 'vi.stackexchange.com.txt', 'video', 'viewport', 'virtual-battlespace', 'virtual-reality', 'visual-novel', 'visual-studio', 'visualization', 'voice', 'voxels', 'vsync', 'vulkan', 'water', 'waypoints', 'web', 'webapps.stackexchange.com.txt', 'webgl', 'webmasters.stackexchange.com.txt', 'websocket', 'wii', 'window-management', 'windows', 'windows-forms', 'windows-phone-7', 'windowsphone.stackexchange.com.txt', 'woodworking.stackexchange.com.txt', 'word-games', 'wordpress.stackexchange.com.txt', 'workplace.stackexchange.com.txt', 'world-of-warcraft-modding', 'worldbuilding.stackexchange.com.txt', 'wpf', 'writers.stackexchange.com.txt', 'xbox', 'xbox360', 'xcode', 'xna', 'xna-4.0', 'zbrush'],
    'twentynewsgroups': ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'],
    # 'twentynewsgroups': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
}

CLASSIFICATION_NAME2LABELS = {
    "AmazonCounterfactualClassification": ['counterfactual', 'not-counterfactual'],
    "AmazonPolarityClassification": ['positive', 'negative'],
    "AmazonReviewsClassification": ['0', '1', '2', '3', '4'],
    "Banking77Classification": ['refund_not_showing_up', 'activate_my_card', 'age_limit', 'apple_pay_or_google_pay', 'atm_support', 'automatic_top_up', 'balance_not_updated_after_bank_transfer', 'balance_not_updated_after_cheque_or_cash_deposit', 'beneficiary_not_allowed', 'cancel_transfer', 'card_about_to_expire', 'card_acceptance', 'card_arrival', 'card_delivery_estimate', 'card_linking', 'card_not_working', 'card_payment_fee_charged', 'card_payment_not_recognised', 'card_payment_wrong_exchange_rate', 'card_swallowed', 'cash_withdrawal_charge', 'cash_withdrawal_not_recognised', 'change_pin', 'compromised_card', 'contactless_not_working', 'country_support', 'declined_card_payment', 'declined_cash_withdrawal', 'declined_transfer', 'direct_debit_payment_not_recognised', 'disposable_card_limits', 'edit_personal_details', 'exchange_charge', 'exchange_rate', 'exchange_via_app', 'extra_charge_on_statement', 'failed_transfer', 'fiat_currency_support', 'get_disposable_virtual_card', 'get_physical_card', 'getting_spare_card', 'getting_virtual_card', 'lost_or_stolen_card', 'lost_or_stolen_phone', 'order_physical_card', 'passcode_forgotten', 'pending_card_payment', 'pending_cash_withdrawal', 'pending_top_up', 'pending_transfer', 'pin_blocked', 'receiving_money', 'request_refund', 'reverted_card_payment?', 'supported_cards_and_currencies', 'terminate_account', 'top_up_by_bank_transfer_charge', 'top_up_by_card_charge', 'top_up_by_cash_or_cheque', 'top_up_failed', 'top_up_limits', 'top_up_reverted', 'topping_up_by_card', 'transaction_charged_twice', 'transfer_fee_charged', 'transfer_into_account', 'transfer_not_received_by_recipient', 'transfer_timing', 'unable_to_verify_identity', 'verify_my_identity', 'verify_source_of_funds', 'verify_top_up', 'virtual_card_not_working', 'visa_or_mastercard', 'why_verify_identity', 'wrong_amount_of_cash_received', 'wrong_exchange_rate_for_cash_withdrawal'],
    "EmotionClassification": ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise'],
    "ImdbClassification": ['positive', 'negative'],
    "MassiveIntentClassification": ['alarm_query', 'alarm_remove', 'alarm_set', 'audio_volume_down', 'audio_volume_mute', 'audio_volume_other', 'audio_volume_up', 'calendar_query', 'calendar_remove', 'calendar_set', 'cooking_query', 'cooking_recipe', 'datetime_convert', 'datetime_query', 'email_addcontact', 'email_query', 'email_querycontact', 'email_sendemail', 'general_greet', 'general_joke', 'general_quirky', 'iot_cleaning', 'iot_coffee', 'iot_hue_lightchange', 'iot_hue_lightdim', 'iot_hue_lightoff', 'iot_hue_lighton', 'iot_hue_lightup', 'iot_wemo_off', 'iot_wemo_on', 'lists_createoradd', 'lists_query', 'lists_remove', 'music_dislikeness', 'music_likeness', 'music_query', 'music_settings', 'news_query', 'play_audiobook', 'play_game', 'play_music', 'play_podcasts', 'play_radio', 'qa_currency', 'qa_definition', 'qa_factoid', 'qa_maths', 'qa_stock', 'recommendation_events', 'recommendation_locations', 'recommendation_movies', 'social_post', 'social_query', 'takeaway_order', 'takeaway_query', 'transport_query', 'transport_taxi', 'transport_ticket', 'transport_traffic', 'weather_query'],
    "MassiveScenarioClassification": ['alarm', 'audio', 'calendar', 'cooking', 'datetime', 'email', 'general', 'iot', 'lists', 'music', 'news', 'play', 'qa', 'recommendation', 'social', 'takeaway', 'transport', 'weather'],
    "MTOPDomainClassification": ['alarm', 'calling', 'event', 'messaging', 'music', 'news', 'people', 'recipes', 'reminder', 'timer', 'weather'],
    "MTOPIntentClassification": [l.upper() for l in ['add_time_timer', 'add_to_playlist_music', 'answer_call', 'cancel_call', 'cancel_message', 'create_alarm', 'create_call', 'create_playlist_music', 'create_reminder', 'create_timer', 'delete_alarm', 'delete_playlist_music', 'delete_reminder', 'delete_timer', 'dislike_music', 'disprefer', 'end_call', 'fast_forward_music', 'follow_music', 'get_age', 'get_airquality', 'get_alarm', 'get_attendee_event', 'get_availability', 'get_call', 'get_call_contact', 'get_call_time', 'get_category_event', 'get_contact', 'get_contact_method', 'get_date_time_event', 'get_details_news', 'get_education_degree', 'get_education_time', 'get_employer', 'get_employment_time', 'get_event', 'get_gender', 'get_group', 'get_info_contact', 'get_info_recipes', 'get_job', 'get_language', 'get_life_event', 'get_life_event_time', 'get_location', 'get_lyrics_music', 'get_major', 'get_message', 'get_message_contact', 'get_mutual_friends', 'get_recipes', 'get_reminder', 'get_reminder_amount', 'get_reminder_date_time', 'get_reminder_location', 'get_stories_news', 'get_sunrise', 'get_sunset', 'get_timer', 'get_track_info_music', 'get_undergrad', 'get_weather', 'help_reminder', 'hold_call', 'ignore_call', 'is_true_recipes', 'like_music', 'loop_music', 'merge_call', 'pause_music', 'pause_timer', 'play_media', 'play_music', 'prefer', 'previous_track_music', 'question_music', 'question_news', 'remove_from_playlist_music', 'repeat_all_music', 'repeat_all_off_music', 'replay_music', 'restart_timer', 'resume_call', 'resume_music', 'resume_timer', 'rewind_music', 'send_message', 'set_available', 'set_default_provider_calling', 'set_default_provider_music', 'set_rsvp_interested', 'set_rsvp_no', 'set_rsvp_yes', 'set_unavailable', 'share_event', 'silence_alarm', 'skip_track_music', 'snooze_alarm', 'start_shuffle_music', 'stop_music', 'stop_shuffle_music', 'subtract_time_timer', 'switch_call', 'unloop_music', 'update_alarm', 'update_call', 'update_method_call', 'update_reminder', 'update_reminder_date_time', 'update_reminder_location', 'update_reminder_todo', 'update_timer']],
    "ToxicConversationsClassification": ['toxic', 'not toxic'],
    "ToxicConversationsPairClassification": ['toxic', 'not toxic'],
    "TweetSentimentExtractionClassification": ['positive', 'negative', 'neutral'],
    "TweetSentimentPairClassification": ['positive', 'negative', 'neutral'],
}

@AutoPrompt.register("sfr-deprecated")
def load_sfe_prompt(task_name, task_type="Retrieval", *args, **kwargs):
    if task_name.endswith("_small") or task_name.endswith("_s") or task_name.endswith("_xs"):
        task_name = task_name[:task_name.rindex("_")]
    if task_name.startswith("cqadupstack-"):
        task_name = "cqadupstack"
    assert task_name in sfr_prompts_map_deprecated, f"Unsupported dataset name {task_name}"
    prompt_dict = copy.copy(sfr_prompts_map_deprecated[task_name.lower()])
    prompt_dict['q_prompt'] += ': '
    prompt_dict['d_prompt'] += ': '
    return prompt_dict


@AutoPrompt.register("sfr")
def load_sfr_prompt(task_name, task_type, add_choices_in_prompt=False, add_taskname_in_docprompt=False, *args, **kwargs):
    if task_type is None:
        task_type = "Retrieval"
    if task_name.endswith("_small") or task_name.endswith("_s") or task_name.endswith("_xs"):
        task_name = task_name[:task_name.rindex("_")]
    if task_name.startswith("cqadupstack-"):
        task_name = "cqadupstack"
    task_def = get_task_def(task_name=task_name, task_type=task_type)
    if add_choices_in_prompt:
        labels = get_labels(task_name=task_name, task_type=task_type)
        if len(labels) <= 20:
            task_def = f"{task_def} Candidate labels ({len(labels)} labels) are: {labels}."
    q_prompt = get_detailed_instruct(task_def)
    d_prompt = ""
    if add_taskname_in_docprompt and task_type == "Retrieval":
        d_prompt = f"[{task_name}]"
    prompt_dict = {"q_prompt": q_prompt, "d_prompt": d_prompt}
    return prompt_dict


def get_task_def(task_type: str, task_name: str) -> str:
    # @ruimeng added
    if task_name.lower() in ['nli', 'allnli']:
        return "Retrieve a sentence that is semantically entailed by the given sentence."

    if task_type in ['STS', 'sts']:
        return "Retrieve semantically similar text."

    if task_type in ['Summarization', 'summarization']:
        return "Given a news summary, retrieve other semantically similar summaries."

    if task_type in ['BitextMining', 'bitextmining']:
        return "Retrieve parallel sentences."

    if task_type in ['Classification', 'classification']:
        task_name_to_instruct: Dict[str, str] = {
            'AmazonCounterfactualClassification': 'Classify a given Amazon customer review text by counterfactual or not-counterfactual.',
            'AmazonPolarityClassification': 'Classify Amazon reviews into positive or negative sentiment.',
            'AmazonReviewsClassification': 'Classify a given Amazon review by its the rating from 0 to 4.',
            'Banking77Classification': 'Given a online banking query, predict the user\'s intent.',
            'EmotionClassification': 'Classify the emotion expressed in a given Twitter message into one of the six emotions.',
            'ImdbClassification': 'Classify the sentiment expressed in a given IMDB movie review.',
            'MassiveIntentClassification': 'Given an Alexa user utterance, predict the intent of the user.',
            'MassiveScenarioClassification': 'Given an Alexa user utterance, predict the task/scenario this utterance implies.',
            'MTOPDomainClassification': 'Given an utterance from a virtual assistant user, classify its intent domain.',
            'MTOPIntentClassification': 'Given an utterance from a virtual assistant user, classify the user\'s intent.',
            'ToxicConversationsClassification': 'Classify a given comment from the Civil Comments platform by its toxicity.',
            'TweetSentimentExtractionClassification': 'Classify a given tweet by its sentiment.',
            # 'AmazonReviewsPairClassification': 'Given an Amazon review, locate reviews within the same rating category.',
            # 'EmotionPairClassification': 'Given an Twitter message, locate message within the same emotion category.',
            # 'MTOPIntentPairClassification': 'Given an utterance in task-oriented conversation, locate utterance within the same intent category.',
            # 'ToxicConversationsPairClassification': 'Given an comment as toxic or non-toxic, locate comments within the same category.',
            # 'TweetSentimentPairClassification': 'Given an comment as either positive, negative, or neutral, locate comments within the same category.',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Clustering', 'clustering']:
        task_name_to_instruct: Dict[str, str] = {
            'ArxivClusteringP2P': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts.',
            'ArxivClusteringS2S': 'Identify the main and secondary category of Arxiv papers based on the titles.',
            'BiorxivClusteringP2P': 'Identify the main category of Biorxiv papers based on the titles and abstracts.',
            'BiorxivClusteringS2S': 'Identify the main category of Biorxiv papers based on the titles.',
            'MedrxivClusteringP2P': 'Identify the main category of Medrxiv papers based on the titles and abstracts.',
            'MedrxivClusteringS2S': 'Identify the main category of Medrxiv papers based on the titles.',
            'RedditClustering': 'Identify the topic or theme of Reddit posts based on the titles.',
            'RedditClusteringP2P': 'Identify the topic or theme of Reddit posts based on the titles and posts.',
            'StackExchangeClustering': 'Identify the topic or theme of StackExchange posts based on the titles.',
            'StackExchangeClusteringP2P': 'Identify the topic or theme of StackExchange posts based on the given paragraphs.',
            'TwentyNewsgroupsClustering': 'Identify the topic or theme of a newsgroups post given its subject/title.',
            'TwentyNewsgroupsClustering-label': 'Predict the topic category of a newsgroups post given its subject/title.',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Reranking', 'PairClassification', 'reranking', 'pairclassification']:
        task_name_to_instruct: Dict[str, str] = {
            # Reranking
            'AskUbuntuDupQuestions': 'Retrieve duplicate questions from AskUbuntu forum.',
            'MindSmallReranking': 'Retrieve relevant news articles based on user browsing history.',
            'SciDocsRR': 'Given a title of a scientific paper, retrieve the titles of other relevant papers.',
            'StackOverflowDupQuestions': 'Retrieve duplicate questions from StackOverflow forum.',
            # PairClassification
            'SprintDuplicateQuestions': 'Retrieve duplicate questions from Sprint forum.',
            'TwitterSemEval2015': 'Retrieve tweets that are semantically similar to the given tweet.',
            'TwitterURLCorpus': 'Retrieve tweets that are semantically similar to the given tweet.',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Retrieval', 'retrieval']:
        if task_name.lower().startswith('cqadupstack'):
            return 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question'

        task_name_to_instruct: Dict[str, str] = {
            'ArguAna': 'Given a claim, find documents that refute the claim.',
            'ClimateFEVER': 'Given a claim about climate change, retrieve documents that support or refute the claim.',
            'DBPedia': 'Given a query, retrieve relevant entity descriptions from DBPedia.',
            'FEVER': 'Given a claim, retrieve documents that support or refute the claim.',
            'FiQA2018': 'Given a financial question, retrieve user replies that best answer the question.',
            'HotpotQA': 'Given a multi-hop question, retrieve documents that can help answer the question.',
            'MSMARCO': 'Given a web search query, retrieve relevant passages that answer the query.',
            'NFCorpus': 'Given a question, retrieve relevant documents that best answer the question.',
            'NQ': 'Given a question, retrieve Wikipedia passages that answer the question.',
            'QuoraRetrieval': 'Given a question, retrieve questions that are semantically equivalent to the given question.',
            'SCIDOCS': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper.',
            'SciFact': 'Given a scientific claim, retrieve documents that support or refute the claim.',
            'Touche2020': 'Given a question, retrieve detailed and persuasive arguments that answer the question.',
            'TRECCOVID': 'Given a query on COVID-19, retrieve documents that answer the query',
            'InstructConversation': "Given a question asked by user, the assistant answers",
            'MrTydi': "Given a question, retrieve Wikipedia passages that answer the question",
            "ChatgptShortLong": "Given a query, retrieve passages that answer the query",
            'mtp': "Given a web search query, retrieve relevant passages that answer the query.",
        }

        # add lower case keys to match some beir names
        task_name_to_instruct.update({k.lower(): v for k, v in task_name_to_instruct.items()})
        # other cases where lower case match still doesn't work
        task_name_to_instruct['trec-covid'] = task_name_to_instruct['TRECCOVID']
        task_name_to_instruct['climate-fever'] = task_name_to_instruct['ClimateFEVER']
        task_name_to_instruct['dbpedia-entity'] = task_name_to_instruct['DBPedia']
        task_name_to_instruct['webis-touche2020'] = task_name_to_instruct['Touche2020']
        task_name_to_instruct['fiqa'] = task_name_to_instruct['FiQA2018']
        task_name_to_instruct['quora'] = task_name_to_instruct['QuoraRetrieval']
        task_name_to_instruct['instructed-conversation'] = task_name_to_instruct['InstructConversation']

        # for miracl evaluation
        task_name_to_instruct['miracl'] = 'Given a question, retrieve Wikipedia passages that answer the question'

        return task_name_to_instruct[task_name]

    raise ValueError(f"No instruction config for task {task_name} with type {task_type}")


def get_labels(task_type: str, task_name: str) -> str:
    if task_type in ['Classification', 'classification']:
        return copy.copy(CLASSIFICATION_NAME2LABELS[task_name])

    if task_type in ['Clustering', 'clustering']:
        cluster_name2labels: Dict[str, str] = {
            'ArxivClusteringP2P': CLUSTERING_NAME2LABELS['arxiv'],
            'ArxivClusteringS2S': CLUSTERING_NAME2LABELS['arxiv'],
            'BiorxivClusteringP2P': CLUSTERING_NAME2LABELS['biorxiv'],
            'BiorxivClusteringS2S': CLUSTERING_NAME2LABELS['biorxiv'],
            'MedrxivClusteringP2P': CLUSTERING_NAME2LABELS['medrxiv'],
            'MedrxivClusteringS2S': CLUSTERING_NAME2LABELS['medrxiv'],
            'RedditClustering': CLUSTERING_NAME2LABELS['reddit'],
            'RedditClusteringP2P': CLUSTERING_NAME2LABELS['reddit'],
            'StackExchangeClustering': CLUSTERING_NAME2LABELS['stackexchange'],
            'StackExchangeClusteringP2P': CLUSTERING_NAME2LABELS['stackexchange'],
            'TwentyNewsgroupsClustering': CLUSTERING_NAME2LABELS['twentynewsgroups'],
        }
        return cluster_name2labels[task_name]

    return []


def get_detailed_instruct(task_description: str) -> str:
    if not task_description:
        return ''

    return 'Instruction: {}\nQuery: '.format(task_description)