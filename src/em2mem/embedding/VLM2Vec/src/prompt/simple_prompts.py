from src.prompt.base_prompt import AutoPrompt

beir_all = ['trec-covid', 'arguana', 'webis-touche2020', 'scidocs', 'scifact', 'nfcorpus', 'fiqa',
            'msmarco', 'dbpedia-entity', 'fever', 'climate-fever', 'nq', 'hotpotqa', 'quora', 'cqadupstack',
            'bioasq', 'signal1m', 'trec-news', 'robust04']

@AutoPrompt.register("none")
def load_dummy_prompt(task_type, task_name, *args, **kwargs):
    return {"q_prompt": "", "d_prompt": ""}


@AutoPrompt.register("e5")
def load_e5_prompt(task_type, task_name, *args, **kwargs):
    return {"q_prompt": "query: ", "d_prompt": "passage: "}
    # TODO note that https://github.com/microsoft/unilm/issues/1325
    # For MTEB evaluation (except its BEIR subset), we simply prepend "query: " to all texts.
    # Also in training, quora task is using "query: " to both query/doc
    # return {"q_prompt": "query: ", "d_prompt": "query: "}


@AutoPrompt.register("bge")
def load_bge_prompt(task_type, task_name="Retrieval", *args, **kwargs):
    if task_name is None or task_name.lower().strip() == "retrieval":
        return {"q_prompt": "Represent this sentence for searching relevant passages: ", "d_prompt": ""}
    else:
        return {"q_prompt": "", "d_prompt": ""}


@AutoPrompt.register("uae")
def load_uae_prompt(task_type, task_name, *args, **kwargs):
    if task_name.lower().startswith('beir') or task_name.lower() in beir_all or task_name.lower() in beir_all:
        return {"q_prompt": "Represent this sentence for searching relevant passages: ", "d_prompt": ""}
    else:
        return {"q_prompt": "", "d_prompt": ""}


@AutoPrompt.register("stella")
def load_stella_prompt(task_type, task_name, *args, **kwargs):
    if task_name.lower().startswith('beir') or task_name.lower() in beir_all or task_name.lower() in beir_all:
        return {"q_prompt": "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: ", "d_prompt": ""}
    else:
        return {"q_prompt": "Instruct: Retrieve semantically similar text.\nQuery: ", "d_prompt": ""}
