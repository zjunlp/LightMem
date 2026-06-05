import json
import sys

import numpy as np

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoProcessor
from src.dataset import EvalDataset
import re

def main():
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    datasets = [
        "GQA",
        # "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R",
        # "ObjectNet", "Country211",
        # "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "VizWiz", "GQA",
        # "TextVQA",
        # "VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA",
        # "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS",
        # "MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"
    ]

    # ToDo: This part of code is a little bit hacky. Need to refactor later.
    for idx, subset in enumerate(datasets):
        eval_qry_dataset = EvalDataset(
            data_args=data_args,
            model_args=model_args,
            subset=subset,
            text_field="qry_text",
            img_path_field="qry_img_path",
        )
        eval_tgt_dataset = EvalDataset(
            data_args=data_args,
            model_args=model_args,
            subset=subset,
            text_field="tgt_text",
            img_path_field="tgt_img_path",
        )
        tgttokens = []
        tgtstr_lens = []
        for tgt in eval_tgt_dataset:
            # print(tgt)
            tokens = re.split('[^a-zA-Z]', tgt[0])
            tgttokens.append(tokens)
            tgtstr_lens.append(len(tokens))
            pass

        print(f'dataset: {subset}')
        print(f'tgt-avg-len: {np.mean(tgtstr_lens)}')
        pass


if __name__ == "__main__":
    main()
