# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
import os

import math
import numpy as np
import json
import random
import datasets

from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES
from src.prompt.base_prompt import AutoPrompt
from src.text_utils.normalize_text import normalize


def ex_dict2str(ex_dict, add_title=True):
    if isinstance(ex_dict, str):
        return ex_dict
    if not isinstance(ex_dict, dict):
        raise NotImplementedError(f"Wrong ex_dict object type, expecting dict: {ex_dict}")
    title, text = ex_dict.get('title', ''), ex_dict['text']
    if title and add_title:
        text = title.rstrip('. ') + '. ' + text
    return text

@add_metainfo_hook
def data_prepare(batch_dict, dataset_name=None, num_hardneg=0,
                 query_group_name=None, query_group_range=None,
                 hardneg_range=None, neg_score_threshold=None,
                 posneg_margin=None, posneg_ratio=None,
                 posbased_filter_range=None, posbased_filter_score=None,
                 prevneg_ratio=0.0,
                 query_prompt='', doc_prompt='', *args, **kwargs):
    add_title = not dataset_name.endswith('_title')
    pos_contexts, queries, pos_docs = [], [], []
    neg_contexts, neg_docs = [], []
    num_neg_cands = []
    dataset_names = []
    for data_idx, text in enumerate(batch_dict['text']):
        try:
            example = json.loads(text)
            assert isinstance(example['pos'], list) and isinstance(example['neg'], list), f'Expect pos/neg is a list of strings or dicts, but got {type(example["pos"])} and {type(example["neg"])}'
            q, pos = example['query'], random.choice(example['pos'])
            if query_group_name:
                q_cands = [q.strip() for q in example['BM25_query'] if q.strip()]
                if len(q_cands) > 0:
                    if query_group_range:
                        q_range = query_group_range.split('-')
                        r_start = int(q_range[0]) if q_range[0] else 0
                        r_end = int(q_range[1]) if q_range[1] else len(q_cands)
                        q = random.choice(q_cands[r_start: r_end])
                    else:
                        q = q_cands[0]
                else:
                    pass
            negs = []
            # process NEGs
            if num_hardneg > 0:
                neg_cands, neg_scores_cands = example['neg'], example['neg_scores']
                # dedup
                # unique_neg_ids = sorted(np.unique(example['neg'], return_index=True)[1])
                # neg_cands = [example['neg'][nid] for nid in unique_neg_ids]
                # neg_scores_cands = [example['neg_scores'][nid] for nid in unique_neg_ids]
                # filter by posbased metrics
                docids_to_filter = []
                if 'dataset' in example and 'arguana' in example['dataset'] and 'neg_docids' in example:
                    # specific to arguana, treating passages from the same doc as positives
                    pos_docid_prefix = example['pos'][0]['doc_id'][: example['pos'][0]['doc_id'].rindex('-')]
                    for docid in example['neg_docids']:
                        if docid.startswith(pos_docid_prefix):
                            docids_to_filter.append(docid)
                if (posbased_filter_range or posbased_filter_score) and 'posbased_neg_docids' in example:
                    if posbased_filter_range:
                        posbased_range = posbased_filter_range.split('-')
                        r_start = int(posbased_range[0]) if posbased_range[0] else 0
                        r_end = int(posbased_range[1]) if posbased_range[1] else len(example['posbased_neg'])
                        docids_to_filter.extend(example['posbased_neg_docids'][r_start: r_end])
                    if posbased_filter_score:
                        for score, docid in zip(example['posbased_neg_scores'], example['posbased_neg_docids']):
                            if score > posbased_filter_score:
                                docids_to_filter.append(docid)
                            else:
                                break
                docids_to_filter = set(docids_to_filter)
                if len(docids_to_filter) > 0:
                    valid_neg_idxs = [neg_idx for neg_idx,neg_docid in enumerate(example['neg_docids']) if neg_docid not in docids_to_filter]
                    neg_cands = [neg_cands[nid] for nid in valid_neg_idxs]
                    neg_scores_cands = [neg_scores_cands[nid] for nid in valid_neg_idxs]
                # print(f"preserved = {len(neg_scores_cands)}/{len(example['neg_docids'])} ")
                # print(f"max posbased_neg_scores= {max(example['posbased_neg_scores'])}")

                neg_score_tuples = [(n,s) for n,s in zip(neg_cands, neg_scores_cands)]
                # filter negs by neg_score_threshold, ignore negs whose scores are larger than neg_score_threshold
                if neg_score_threshold is not None:
                    neg_score_tuples = [t for t in neg_score_tuples if t[1] < neg_score_threshold]
                # filter negs by posneg_margin, ignore negs that have scores less than (pos_score-margin)
                if posneg_margin is not None and 'pos_score' in example:
                    pos_score = example['pos_score']
                    neg_score_tuples = [t for t in neg_score_tuples if t[1] < pos_score-posneg_margin]
                # filter negs by posneg_ratio, ignore negs that have scores larger than (pos_score*posneg_ratio)
                if posneg_ratio is not None and 'pos_score' in example:
                    pos_score = example['pos_score']
                    neg_score_tuples = [t for t in neg_score_tuples if t[1] < pos_score*posneg_ratio]
                neg_cands = [t[0] for t in neg_score_tuples]
                if len(neg_cands) == 0:
                    if 'prev_neg' in example:
                        neg_cands = example['prev_neg']
                    else:
                        continue
                # filter by hardneg_range
                if hardneg_range and len(neg_cands) > 0:
                    range_start, range_end = hardneg_range.split('-')
                    range_start = int(range_start) if range_start else 0
                    range_end = int(range_end) if range_end else len(neg_cands)
                    neg_cands = neg_cands[range_start: range_end]
                # skip if not enough neg candidates for sampling
                if len(neg_cands) < num_hardneg * 0.8:
                    continue
                # sample negs from candidates
                if len(neg_cands) < num_hardneg:
                    negs = np.random.choice(neg_cands, size=num_hardneg, replace=True)
                else:
                    negs = np.random.choice(neg_cands, size=num_hardneg, replace=False)
                # add prev_neg, replace the first k negs
                if prevneg_ratio > 0 and random.random() < prevneg_ratio and 'prev_neg' in example and len(example['prev_neg']):
                    prev_negs = np.random.choice(example['prev_neg'], size=min(len(example['prev_neg']), math.ceil(num_hardneg * prevneg_ratio)),
                                                 replace=len(example['prev_neg']) < num_hardneg * prevneg_ratio)
                    for prev_neg_idx, prev_neg in enumerate(prev_negs):
                        prev_neg = ex_dict2str(prev_neg, add_title)
                        negs[prev_neg_idx] = prev_neg
            else:
                negs, neg_cands, neg_scores_cands = [], [], []

            # add Q/POSs: add prompt and normalize the text
            q = ex_dict2str(q, add_title)
            pos = ex_dict2str(pos, add_title)
            q = normalize(query_prompt + q)
            pos = normalize(doc_prompt + pos)
            queries.append(q)
            pos_docs.append(pos)
            pos_contexts.append(text)
            dataset_names.append(dataset_name)
            num_neg_cands.append(len(neg_cands))

            # add NEGs
            if len(negs) > 0:
                _negs = []
                for neg in negs:
                    neg = ex_dict2str(neg, add_title)
                    neg = normalize(doc_prompt + neg)
                    _negs.append(neg)
                neg_docs.append(_negs)
            pass
        except Exception as e:
            print('Error in processing BEIR dataset')
            print(e)
            print(text)
            raise e
            # return {'queries': [[]], 'docs': [[]]}
    batch_len = len(queries)
    return_dict= {"query_text": queries, "query_image": [None] * batch_len,
                  "pos_text": pos_docs, "pos_image": [None] * batch_len,
                  "neg_text": neg_docs if neg_docs else [None] * batch_len, "neg_image": [None] * batch_len}
    return return_dict


DATASET_PARSER_NAME="beir"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_beir_dataset(model_args, data_args, training_args, dataset_name, file_path,
                      query_prompt_type='e5mistral', doc_prompt_type='e5mistral', *args, **kwargs):
    # set up prompt
    prompt_name = kwargs.pop("prompt_name") if "prompt_name" in kwargs else dataset_name
    query_prompt, doc_prompt = "", ""
    if query_prompt_type:
        query_prompt = AutoPrompt.instantiate(prompt_family=query_prompt_type, task_name=prompt_name, **kwargs)['q_prompt']
    if doc_prompt_type:
        doc_prompt = AutoPrompt.instantiate(prompt_family=doc_prompt_type, task_name=prompt_name, **kwargs)['d_prompt']

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{dataset_name}'
    # create dataset
    dataset = datasets.load_dataset("text", data_files=file_path, split='train', keep_in_memory=False)
    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        num_rows = int(num_sample_per_subset)
        dataset = dataset.select(range(num_rows))
    num_rows = dataset.num_rows

    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)  # convert to IterableDataset and multiple shards
    num_hardneg = kwargs.get("num_hardneg", data_args.num_hardneg)
    dataset = dataset.map(lambda x: data_prepare(x, dataset_name=dataset_name, num_hardneg=num_hardneg,
                                                 query_prompt=query_prompt, doc_prompt=doc_prompt,
                                                 **kwargs),
                          drop_last_batch=True,
                          batched=True, batch_size=1024,
                          remove_columns=['text'])
    # dataset = dataset.shuffle(buffer_size=1024*8*1, seed=42)

    # num_rows in iterable_dataset is not available, set it here for printing dataset stats
    setattr(dataset, 'num_rows', num_rows)

    return dataset
