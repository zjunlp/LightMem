# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
import copy
import json
import os
from collections import defaultdict

import numpy as np
import random
import datasets

from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook
from src.prompt.base_prompt import AutoPrompt
from src.text_utils.normalize_text import normalize
from src.prompt.sfr import CLASSIFICATION_NAME2LABELS
from src.text_utils.basic_utils import print_rank, print_master

SEP = '\t'

@add_metainfo_hook
def data_prepare(examples, dataset_name,
                 query_prompt='', doc_prompt='', num_hardneg=0,
                 local_load=False, **kwargs):
    """
    data_type=against_text is only applicable for classification where a label is assigned as target
    """
    contexts, queries, pos_docs = [], [], []
    neg_contexts, neg_docs = [], []
    ex_ids = []
    all_labels = CLASSIFICATION_NAME2LABELS[dataset_name] if dataset_name in CLASSIFICATION_NAME2LABELS else []
    data_type = kwargs.get("data_type", "default")  # legacy

    # in case of against_text mode, we first gather examples by labels
    label2texts, label2negtexts = defaultdict(list), defaultdict(list)
    if data_type == "against_text":
        for data_idx, text in enumerate(examples['text']):
            if local_load:
                example = json.loads(text)
                text = example["query"] if "query" in example else example["text"]
                pos = example["pos"].lower().strip() if "pos" in example else example["label_text"]
            else:
                text = examples['text'][data_idx]
                pos = examples['label_text'][data_idx].lower().strip()
            label2texts[pos].append(text)
        for pos, texts in label2texts.items():
            for neg_l in label2texts.keys():
                if neg_l != pos:
                    label2negtexts[neg_l].extend(texts)
        for pos in label2texts.keys():
            # since later we only sample a slice, so we shuffle the whole list first
            random.shuffle(label2negtexts[pos])

    # print_rank(f"dataset={kwargs.get('global_dataset_name')}, #data={len(examples['text'])}, #labels={len(label2texts)}")
    for data_idx, text in enumerate(examples['text']):
        try:
            if local_load:
                example = json.loads(text)
                text = example['query'] if "query" in example else example["text"]
                # it's labels in case of classification/clustering
                pos = example['pos'].lower().strip() if "pos" in example else example["label_text"]
                label = copy.copy(pos)
                negs = example["neg"] if "neg" in example else []
            else:
                text = examples['text'][data_idx]
                pos = examples['label_text'][data_idx].lower().strip()
                label = examples['label_text'][data_idx].lower().strip()
                negs = []
            text = normalize(query_prompt + text)
            # In case of against_text and classification, need to sample a pos text and all texts use query_prompt
            if data_type == "against_text":
                pos = random.choice(label2texts[label])
                pos = normalize(query_prompt + pos)
            else:
                pos = normalize(doc_prompt + pos)
            # add NEGs
            if num_hardneg > 0:
                if len(negs) > 0:
                    # reranking/sts, where negs are given
                    negs = np.random.choice(negs, size=num_hardneg, replace=len(negs) < num_hardneg)
                    negs = [normalize(doc_prompt + neg) for neg in negs]
                else:
                    # classification
                    if data_type == "against_text":
                        # sample negs from texts of other labels
                        neg_texts = label2negtexts[label]
                        # use a random slice to reduce the sampling time
                        _start = random.randint(0, len(neg_texts) - num_hardneg * 2)
                        _neg_texts = neg_texts[_start: _start+num_hardneg * 2]
                        negs = np.random.choice(_neg_texts, size=num_hardneg, replace=len(_neg_texts) < num_hardneg)
                        negs = [normalize(query_prompt + neg) for neg in negs]
                    else:
                        negs = [l for l in all_labels if l != pos]
                        negs = np.random.choice(negs, size=num_hardneg, replace=len(negs) < num_hardneg)
                        negs = [normalize(doc_prompt + neg) for neg in negs]
                neg_docs.append(negs)
            contexts.append(text)
            queries.append(text)
            pos_docs.append(pos)
            if "id" in example:
                ex_ids.append(example["id"])
        except Exception as e:
            print(f'Error in processing {dataset_name} data, id={id}')
            # print(e)
            raise e

    # return_dict = {'contexts': contexts, 'queries': queries, 'docs': pos_docs}
    # if num_hardneg > 0:
    #     return_dict['neg_docs'] = neg_docs
    #     return_dict['neg_contexts'] = contexts
    batch_len = len(queries)
    return_dict = {"query_text": queries, "query_image": [None] * batch_len,
                   "pos_text": pos_docs, "pos_image": [None] * batch_len,
                   "neg_text": neg_docs if neg_docs else [None] * batch_len, "neg_image": [None] * batch_len}
    return return_dict

DATASET_PARSER_NAME = "mteb_training"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_mteb_training(model_args, data_args, training_args, dataset_name,
               file_path=None,
              query_prompt_type='e5mistral', doc_prompt_type='e5mistral',
              *args, **kwargs):
    prompt_name = kwargs.pop("prompt_name") if "prompt_name" in kwargs else dataset_name
    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{dataset_name}'
    kwargs['model_backbone'] = model_args.model_backbone
    query_prompt, doc_prompt = '', ''
    if query_prompt_type:
        query_prompt = AutoPrompt.instantiate(prompt_family=query_prompt_type, task_name=prompt_name, **kwargs)["q_prompt"]
    if doc_prompt_type:
        doc_prompt = AutoPrompt.instantiate(prompt_family=doc_prompt_type, task_name=prompt_name, **kwargs)["d_prompt"]
    if os.path.isfile(file_path):
        dataset = datasets.load_dataset("text", split="train", data_files=file_path, keep_in_memory=False, streaming=False)
        num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
        if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
            num_rows = int(num_sample_per_subset)
            dataset = dataset.select(range(num_rows))
        num_rows = dataset.num_rows
        remove_columns = ["text"]
    else:
        raise NotImplementedError("not tested yet")
        # load from HF hub
        config_name = "en" if "en" in datasets.get_dataset_config_names(file_path) else None
        # dataset = datasets.load_dataset(file_path, split="train", name=config_name, keep_in_memory=False, streaming=True, trust_remote_code=True)
        dataset = datasets.load_dataset(file_path, split="test", name=config_name, keep_in_memory=False, streaming=True, trust_remote_code=True)
        remove_columns = dataset.column_names
        print_master(dataset.column_names)
        # some datasets don't have id column and it causes problems
        if remove_columns and "id" in remove_columns:
            dataset = dataset.remove_columns(["id"])
        print_master(dataset.column_names)
    # print_rank(f"Initializing data loader for {dataset_name}")

    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)  # convert to IterableDataset and multiple shards

    # MUST USE datasets == 2.16.1 and fsspec==2023.9.2
    # large batch size will cause HF.datasets fail to return Feature, then the whole dataset will be ignored
    # buffer_size = 1024 * 100 if "amazonreviews" in dataset_name.lower() else 1024 * 64
    # batch_size = 1024 * 80 if "amazonreviews" in dataset_name.lower() else 1024 * 32
    num_hardneg = kwargs.get("num_hardneg", data_args.num_hardneg)
    dataset = dataset.map(lambda x: data_prepare(x, num_hardneg=num_hardneg,
                                                 dataset_name=dataset_name,
                                                 query_prompt=query_prompt, doc_prompt=doc_prompt,
                                                 local_load=os.path.isfile(file_path),
                                                 **kwargs),
                          drop_last_batch=True,
                          batched=True, batch_size=512*4,  # batch size has to be smaller than each data size due to bugs in datasets>=2.19.2
                          remove_columns=remove_columns)
    # buffer_size = 1024 * 32
    # dataset = dataset.shuffle(buffer_size=buffer_size, seed=configs.seed)
    # dataset = dataset._resolve_features()
    # num_rows in iterable_dataset is overridden, set it here for printing dataset stats
    setattr(dataset, 'num_rows', num_rows)

    return dataset
