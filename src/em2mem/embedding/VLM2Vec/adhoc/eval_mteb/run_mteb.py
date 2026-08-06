import copy
import torch
import torch.distributed as dist

import tqdm
import numpy as np
import os

from functools import partial
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from mteb import MTEB

from adhoc.eval_mteb.e5mistral_prompt import load_e5mistral_prompt
from src.arguments import ModelArguments, DataArguments, TrainingArguments, MTEBArguments
from transformers import HfArgumentParser, AutoTokenizer

from src.model.model import MMEBModel
from adhoc.eval_mteb.mteb_utils import logger, pool, move_to_cuda, input_transform_func, varsize_gather_nograd, is_main, str2bool
from src.model.processor import load_processor

# (not effective here, add them in environment variables) for clustering: OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata.
default_n_threads = 1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"


MTEB_TASKS_EN = [
    "AmazonCounterfactualClassification", "AmazonPolarityClassification", "AmazonReviewsClassification", "Banking77Classification", "EmotionClassification", "ImdbClassification", "MassiveIntentClassification", "MassiveScenarioClassification", "MTOPDomainClassification", "MTOPIntentClassification", "ToxicConversationsClassification", "TweetSentimentExtractionClassification",
    "ArxivClusteringP2P", "ArxivClusteringS2S", "BiorxivClusteringP2P", "BiorxivClusteringS2S", "MedrxivClusteringP2P", "MedrxivClusteringS2S", "RedditClustering", "RedditClusteringP2P", "StackExchangeClustering", "StackExchangeClusteringP2P", "TwentyNewsgroupsClustering",
    "SprintDuplicateQuestions", "TwitterSemEval2015", "TwitterURLCorpus",
    "AskUbuntuDupQuestions", "MindSmallReranking", "SciDocsRR", "StackOverflowDupQuestions",
    "ArguAna", "ClimateFEVER", "CQADupstackAndroidRetrieval", "DBPedia", "FEVER", "FiQA2018", "HotpotQA", "MSMARCO", "NFCorpus", "NQ", "QuoraRetrieval", "SCIDOCS", "SciFact", "TRECCOVID", "Touche2020",
    "BIOSSES", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS16", "STS17", "STS22", "STSBenchmark",
    "SummEval"
]


class DenseEncoder(torch.nn.Module):
    def __init__(self, model_args, mteb_args, max_length=512, **kwargs):
        super().__init__()
        self.max_length = max_length
        self.pool_type = model_args.pooling

        processor = load_processor(model_args)
        model = MMEBModel.load(model_args)

        processor.tokenizer.padding_side = "right"
        model.eval()
        model = model.to(mteb_args.device, dtype=torch.bfloat16)
        self.encoder = model
        self.tokenizer = processor.tokenizer
        self.processor = processor

        self.batch_size_per_device = mteb_args.batch_size_per_device
        self.gpu_count = torch.cuda.device_count()
        self.encoder.eval()
        self.encoder.cuda()
        self.query_prompt = ""
        self.doc_prompt = ""
        self.sep = ". "

        if not torch.distributed.is_initialized() and self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

    def encode_queries(self, sentences, **kwargs) -> np.ndarray:
        return self.encode(sentences, self.query_prompt, is_query=True, **kwargs)

    def encode_corpus(self, sentences, **kwargs) -> np.ndarray:
        return self.encode(sentences, self.doc_prompt, is_query=False, **kwargs)

    @torch.no_grad()
    def encode(self, inputs, prompt=None, is_query=True, **kwargs) -> np.ndarray:
        """ Returns a list of embeddings for the given sentences.
        Args:
            inputs (`List[str]`): List of sentences to encode
            batch_size_per_device (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        if isinstance(inputs[0], dict):
            input_texts = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in inputs]
        else:
            input_texts = copy.copy(inputs)
        if torch.distributed.is_initialized() and len(input_texts) >= dist.get_world_size():
            idx = np.array_split(range(len(input_texts)), dist.get_world_size())[dist.get_rank()]
        else:
            # in case of non-DDP or not enough sentences, all devices are running the same job, but no gathering in the end
            idx = range(len(input_texts))
        device_sentences = [input_texts[i] for i in idx]
        # for tasks other than RET
        if is_query and not prompt and self.query_prompt:
            prompt = self.query_prompt
        if prompt:
            device_sentences_with_prompt = [prompt + (s['text'] if isinstance(s, dict) else s) for s in device_sentences]
        else:
            device_sentences_with_prompt = device_sentences

        dataset: Dataset = Dataset.from_dict({'input_texts': device_sentences_with_prompt})
        dataset.set_transform(partial(input_transform_func, self.tokenizer, max_length=self.max_length, always_add_eos=True))
        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=1)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size_per_device if torch.distributed.is_initialized() else self.batch_size_per_device * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=data_collator,
            pin_memory=True)

        encoded_embeds = []
        # for batch in data_loader:
        for batch in tqdm.tqdm(data_loader, desc="encoding", miniters=10, disable=not is_main()):
            # batch.data['is_causal'] = self.is_causal  # only needed for Qwen
            # print(f"batch.data['is_causal']={batch.data['is_causal']}")
            # print(self.tokenizer.decode(batch['input_ids'][0]))
            # print(batch['input_ids'].numpy())
            # print(batch)
            batch = move_to_cuda(batch)
            with torch.cuda.amp.autocast():
                outputs = self.encoder.encode_input(batch)
                encoded_embeds.append(outputs)
        encoded_embeds = torch.cat(encoded_embeds, dim=0)
        if torch.distributed.is_initialized() and len(inputs) >= dist.get_world_size():
            encoded_embeds = varsize_gather_nograd(encoded_embeds)
        encoded_embeds = encoded_embeds.cpu().numpy()

        return encoded_embeds

    def set_prompt(self, query_prompt: str, doc_prompt: str):
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, MTEBArguments, TrainingArguments))
    model_args, data_args, mteb_args, training_args, remaining_args  = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    model_args: ModelArguments
    data_args: DataArguments
    mteb_args: MTEBArguments

    assert mteb_args.eval_output_dir, 'eval_output_dir should be specified'
    os.makedirs(mteb_args.eval_output_dir, exist_ok=True)

    task_types = None
    tasks = ['NFCorpus', 'FiQA2018', 'ArguAna', 'SciFact', 'SCIDOCS', 'Touche2020', 'TRECCOVID']
    # tasks = ["BiorxivClusteringS2S", "MedrxivClusteringS2S", "RedditClustering", "StackExchangeClustering", "StackExchangeClusteringP2P", "TwentyNewsgroupsClustering"]
    evaluation = MTEB(task_types=task_types, tasks=tasks, task_langs=["eng-Latn", "en"])
    model = DenseEncoder(model_args, mteb_args, max_length=mteb_args.max_length)

    for task_cls in evaluation.tasks:
        task_name: str = task_cls.metadata.name
        task_type: str = task_cls.metadata.type
        # filter out not supported datasets
        print(f"Evaluating MTEB: {task_type} - {task_name}")
        # filter out not supported datasets
        if task_name not in MTEB_TASKS_EN:
            continue

        eval_splits = task_cls.metadata.eval_splits
        if "test" not in eval_splits:
            logger.warning("Test split not found for task: {}, type: {}, eval_splits: {}".format(task_name, task_type, eval_splits))
        eval_splits = ["test" if "test" in eval_splits else eval_splits[0]]

        if mteb_args.prompt_family:
            prompt_data = load_e5mistral_prompt(prompt_family=mteb_args.prompt_family, task_name=task_name, task_type=task_type)
            query_prompt = prompt_data['q_prompt']
            doc_prompt = prompt_data['d_prompt']
            model.set_prompt(query_prompt=query_prompt, doc_prompt=doc_prompt)
            logger.info('Set prompt: query={}, doc={}'.format(query_prompt, doc_prompt))
        else:
            logger.info('No prompt is set')

        # disable l2 normalize for classification tasks, as it achieves slightly better results
        if task_type == 'Classification':
            logger.info('Set l2_normalize to False for classification task')
            model.l2_normalize = False
        else:
            model.l2_normalize = True
            logger.info('Set l2_normalize to {}'.format(model.l2_normalize))

        sub_eval = MTEB(tasks=[task_name], task_langs=["eng-Latn", "en"], n_experiments=1)
        logger.info('Running evaluation for task: {}, type: {}'.format(task_name, task_type))
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized():
            mteb_result_folder = mteb_args.eval_output_dir
        else:
            mteb_result_folder = None
        sub_eval.run(
            model, eval_splits=eval_splits,
            output_folder=mteb_result_folder
        )


if __name__ == '__main__':
    main()