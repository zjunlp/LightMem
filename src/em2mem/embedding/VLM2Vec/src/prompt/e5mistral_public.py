import copy
import random
from typing import Dict

from src.prompt.base_prompt import AutoPrompt


@AutoPrompt.register("e5mistral")
def load_e5mistral_prompt(task_name, task_type, *args, **kwargs):
    if task_type is None:
        task_type = "Retrieval"
    if task_name.endswith("_small") or task_name.endswith("_s") or task_name.endswith("_xs"):
        task_name = task_name[:task_name.rindex("_")]
    if task_name.startswith("cqadupstack-"):
        task_name = "cqadupstack"
    task_def = get_task_def_by_task_name_and_type(task_name=task_name, task_type=task_type)
    prompt = get_detailed_instruct(task_def)
    prompt_dict = {"q_prompt": prompt, "d_prompt": ""}
    return prompt_dict


def get_task_def_by_task_name_and_type(task_name: str) -> str:
    # https://arxiv.org/pdf/2401.00368 Table 13
    # https://arxiv.org/pdf/2404.05961 Table 8
    if task_name.lower() in ['nli', 'allnli']:
        return random.choice(["Given a premise, retrieve a hypothesis that is entailed by the premise.",
                              "Find questions that have the same meaning as the input question",])

    if task_name.lower() in ['nli', 'allnli']:
        return "Retrieve semantically similar text."

    '''
    DuReader Given a Chinese search query, retrieve web passages that answer the question
    ELI5 Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum
    FEVER Given a claim, retrieve documents that support or refute the claim
    HotpotQA Given a multi-hop question, retrieve documents that can help answer the question
    MIRACL Given a question, retrieve Wikipedia passages that answer the question
    MrTyDi Given a question, retrieve Wikipedia passages that answer the question
    MSMARCO Passage Given a web search query, retrieve relevant passages that answer the query
    MSMARCO Document Given a web search query, retrieve relevant documents that answer the query
    NQ Given a question, retrieve Wikipedia passages that answer the question
    QuoraDuplicates Given a question, retrieve questions that are semantically equivalent to the given question
    Find questions that have the same meaning as the input question
    SQuAD Retrieve Wikipedia passages that answer the question
    T2Ranking Given a Chinese search query, retrieve web passages that answer the question
    TriviaQA Retrieve Wikipedia passages that answer the question
    '''


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
            "ChatgptShortLong": "Given a query, retrieve passages that answer the query"
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


def get_detailed_instruct(task_description: str) -> str:
    if not task_description:
        return ''

    return 'Instruct: {}\nQuery: '.format(task_description)