# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import json
import os
import random

import datasets
import numpy as np

from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook
from src.prompt.base_prompt import AutoPrompt
from src.text_utils.normalize_text import normalize

SEP = ':\t'
@add_metainfo_hook
def data_prepare(examples,
                 dataset_name='berri',
                 num_hardneg=0, add_prompt_ratio=1.0,
                 query_prompt='', doc_prompt='', **kwargs):
    contexts, queries, docs = [], [], []
    neg_contexts, neg_docs = [], []
    dataset_names = []
    for data_idx, text in enumerate(examples['text']):
        try:
            example = json.loads(text)
            q = example['question']
            q_prompt, q = q.split('[SEP]')
            ans = example['answers']
            pos_list = example['positive_ctxs']
            neg_cands = example['hard_negative_ctxs']
            randomneg_list = example['negative_ctxs']
            if len(neg_cands) == 0 or num_hardneg > len(neg_cands):
                neg_cands = neg_cands + randomneg_list
            pos = random.choice(pos_list)
            # 'title' can be actual title, prompt (web paragraph, pubmed etc.) or null
            pos_prompt, pos = pos['title'], pos['text']
            # override prompt if it's given outside
            if query_prompt: q_prompt = query_prompt
            if doc_prompt: pos_prompt = doc_prompt
            add_prompt = random.random() < add_prompt_ratio
            if add_prompt:
                q = q_prompt + SEP + q if q_prompt else q
                pos = pos_prompt + SEP + pos if pos_prompt else pos
            q = normalize(q)
            pos = normalize(pos)

            # add NEGs
            if len(neg_cands) < num_hardneg:
                negs = np.random.choice(neg_cands, size=num_hardneg, replace=True)
            else:
                negs = np.random.choice(neg_cands, size=num_hardneg, replace=False)
            if len(negs) > 0:
                _negs = []
                for neg in negs:
                    neg_prompt, neg = neg['title'], neg['text']
                    if doc_prompt: neg_prompt = doc_prompt
                    if add_prompt:
                        neg = neg_prompt + SEP + neg if neg_prompt else neg
                    neg = normalize(neg)
                    _negs.append(neg)
                neg_docs.append(_negs)
                neg_contexts.append(text)
            contexts.append(text)
            queries.append(q)
            docs.append(pos)
            dataset_names.append(dataset_name)
        except Exception as e:
            print('Error in processing text to D/Q')
            print(e)
            print(text)
            raise e

    batch_len = len(queries)
    return_dict = {"query_text": queries, "query_image": [None]*batch_len,
                   "pos_text": docs, "pos_image": [None]*batch_len,
                   "neg_text": neg_docs if neg_docs else [None]*batch_len, "neg_image": [None]*batch_len}
    return return_dict


DATASET_PARSER_NAME = "berri"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_berri(model_args, data_args, training_args,
               file_path=None,
               dataset_name="berri",
               query_prompt_type='e5mistral', doc_prompt_type='e5mistral', *args, **kwargs):
    assert os.path.isfile(file_path), f'{file_path} does not exist.'

    # only use berri prompt in BERRI
    query_prompt, doc_prompt = '', ''
    add_prompt_ratio = kwargs.pop("add_prompt_ratio") if "add_prompt_ratio" in kwargs else 1.0
    kwargs['global_dataset_name'] = DATASET_PARSER_NAME
    kwargs['model_backbone'] = model_args.model_backbone
    query_prompt, doc_prompt = "", ""
    if query_prompt_type:
        query_prompt = AutoPrompt.instantiate(prompt_family=query_prompt_type, task_name=dataset_name)['q_prompt']
    if doc_prompt_type:
        doc_prompt = AutoPrompt.instantiate(prompt_family=doc_prompt_type, task_name=dataset_name)['d_prompt']
    dataset = datasets.load_dataset("text", data_files=file_path, split='train', keep_in_memory=False)
    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        num_rows = int(num_sample_per_subset)
        dataset = dataset.select(range(num_rows))
    num_rows = dataset.num_rows

    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)  # convert to IterableDataset and multiple shards

    num_hardneg = kwargs.get("num_hardneg", data_args.num_hardneg)
    dataset = dataset.map(lambda x: data_prepare(x, num_hardneg=num_hardneg,
                                                 add_prompt_ratio=add_prompt_ratio,
                                                 query_prompt=query_prompt, doc_prompt=doc_prompt,
                                                 **kwargs),
                          drop_last_batch=True,
                          batched=True, batch_size=1024)
    # dataset = dataset.shuffle(buffer_size=1024*16, seed=42)
    # num_rows in iterable_dataset is not available, set it here for printing dataset stats
    setattr(dataset, 'num_rows', num_rows)

    return dataset

